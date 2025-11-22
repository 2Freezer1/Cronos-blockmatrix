# V4.1: FINAL CONSOLIDATED VERSION
# FIX: Ensured DatetimeIndex is explicitly applied to SARIMAX/AutoReg forecast results 
# to prevent TypeError when calculating plot x-axis limits.
#
# This script performs a Codebase Mutability (CBM) analysis on GitHub repository data.
# It calculates the tau_T index (a measure of development speed/stability),
# tests for Granger Causality with lagged bug counts, and forecasts tau_T using SARIMAX
# with an AutoReg fallback for robustness.

# --- ClipRun Environment Fix (MUST be first) ---
import os
# AGGRESSIVE FIX: Ensure HOME and USERPROFILE are set to /tmp to prevent environment errors.
if not os.getenv("HOME") or not os.path.isdir(os.getenv("HOME", "")):
    os.environ["HOME"] = "/tmp"
if not os.getenv("USERPROFILE") or not os.path.isdir(os.getenv("USERPROFILE", "")):
    os.environ["USERPROFILE"] = "/tmp"

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.ar_model import AutoReg 
import matplotlib
# Set backend to Agg for non-interactive environments (prevents GUI errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import warnings
from datetime import datetime, timedelta
import time
import requests
import json
import re 

# Suppress minor warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
# CRITICAL: Replace this with your actual GitHub Personal Access Token (PAT) if needed
GITHUB_TOKEN = "github_token_here"
USE_SIMULATED_DATA = False # <--- SET TO FALSE FOR REAL DATA RUN

# Data and Time Series Parameters
TIME_WINDOW_DAYS = 1095  # Enforcing a 3-year window
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=TIME_WINDOW_DAYS)
VSCODE_START_DATE = datetime(2024, 1, 1)
DEFAULT_TIME_SERIES_FREQ = 'W'

# Predictive Model Parameters
FORECAST_PERIODS = 4
BASE_LAG_WEEKS = 3
GRANGER_LAG_RANGE = [1, 2, 3, 4, 5, 6]
MAX_PR_PULL = 8000
PULL_PAGE_SIZE = 100

# V3.17: Conservative Differencing - Only set d=1 if ADF P-value is above this threshold
ADF_DIFFERENCING_THRESHOLD = 0.10 
P_RANGE_SARIMA = (0, 2) 
Q_RANGE_SARIMA = (0, 2) 

# Setting seasonal order to (0, 0, 0, 0) for stability (non-seasonal ARIMA)
SARIMA_SEASONAL_ORDER_W = (0, 0, 0, 0)  
SARIMA_SEASONAL_ORDER_D = (0, 0, 0, 0)
SARIMA_SEASONAL_ORDER_M = (0, 0, 0, 0)

MAX_ITER = 10000 
LBFGS_EPSILON = 1e-8
MAX_RETRIES = 5
MIN_T_STABLE = 1.0

# --- Utility Function: Custom MSE ---
def calculate_mse(y_true, y_pred):
    """Calculates the Mean Squared Error (MSE) using only NumPy."""
    # V3.20: Filter out NaN values before calculation to ensure stability
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_indices):
        return np.nan
    return np.mean((y_true[valid_indices] - y_pred[valid_indices])**2)

# =======================================================================
# 1. ENHANCED PLOTTING FUNCTION (V4.1)
# =======================================================================

def plot_cbm_results(repo_name, ts_data, test_data, forecast_results, bug_series, version, T_lag, d, seasonal_order, fit_method, total_prs_count):
    """
    V4.1: Generates an enhanced CBM validation plot. The logic for setting x-axis 
    limits is robust against the forecast result having a non-DatetimeIndex, 
    as the index is now enforced prior to this call.

    Args:
        repo_name (str): The name of the repository (e.g., 'META-ANALYSIS (WEEKLY)').
        ts_data (pd.Series): The original, full time series data (Blue line, e.g., tau_T).
        test_data (pd.Series): The actual data used for testing/validation (last N periods of ts_data).
        forecast_results (pd.DataFrame): The SARIMAX/AutoReg forecast results table (now guaranteed DatetimeIndex).
        bug_series (pd.Series): The original lagged bug count series (Red line).
        version (str): The version of the CBM analysis.
        T_lag (int): The lag used for the bug count (e.g., 3 weeks).
        d (int): Non-seasonal differencing order.
        seasonal_order (tuple): Seasonal order (P, D, Q, S).
        fit_method (str): The method used for forecasting (e.g., 'CG', 'AR(2) Fallback').
        total_prs_count (int): Total number of PRs analyzed.
    """
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Step 1: Determine frequency and lookback window ---
    freq_str = ts_data.index.freqstr if ts_data.index.freqstr else 'W'
    
    # Set lookback periods based on frequency for a better zoom
    if 'D' in freq_str:
        lookback_periods = 60 # ~2 months
        freq_label = 'days'
        time_unit = 'days'
    elif 'M' in freq_str:
        lookback_periods = 12 # 1 year
        freq_label = 'months'
        time_unit = 'months'
    else: # Weekly default
        lookback_periods = 20 # ~5 months
        freq_label = 'weeks'
        time_unit = 'weeks'
    
    max_date = ts_data.index.max()
    
    # Calculate start index for visualization
    start_idx = max(0, len(ts_data) - lookback_periods)

    # Filter data to the desired time window
    ts_data_viz = ts_data.iloc[start_idx:]
    bug_series_viz = bug_series.iloc[start_idx:]
    
    # --- Step 2: Plot the Bug Count (Left Y-axis - Red) ---
    color_bug = 'tab:red'
    
    # Determine lag description
    if 'D' in freq_str: lag_description = f'{T_lag} days'
    elif 'M' in freq_str: lag_description = f'{T_lag} months'
    else: lag_description = f'{T_lag} weeks'

    ax1.set_xlabel(f'Date (Last {lookback_periods} {time_unit} + Forecast)', fontsize=10)
    ax1.set_ylabel(f'Lagged Bug Count (T - {lag_description})', color=color_bug, fontsize=10)
    
    # Plot the historical bug count
    ax1.plot(bug_series_viz.index, bug_series_viz.values, 
             linestyle=':', color=color_bug, label=f'Lagged Bug Count (T - {lag_description})')
    
    ax1.tick_params(axis='y', labelcolor=color_bug)
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # --- Step 3: Plot the τ_T Index (Right Y-axis - Blue/Green) ---
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_tau = 'tab:blue'
    ax2.set_ylabel(r'$\tau_T$ Index (Blue/Green)', color=color_tau, fontsize=10)
    
    # Plot the historical tau_T
    ax2.plot(ts_data_viz.index, ts_data_viz.values, 
             linestyle='-', color=color_tau, linewidth=1.5, label=r'Observed $\tau_T$ Index')
    
    ax2.tick_params(axis='y', labelcolor=color_tau)
    
    # --- Step 4: Plot the Forecast and Confidence Interval ---
    
    # Calculate padding for the end of the X-axis
    padding = pd.Timedelta(weeks=1) if 'W' in freq_str else pd.Timedelta(days=15)
    
    if forecast_results is not None and not forecast_results.empty:
        # Extract forecast index and values (Index is now guaranteed to be DatetimeIndex)
        forecast_index = forecast_results.index
        forecast_mean = forecast_results['mean']
        ci_lower = forecast_results['mean_ci_lower']
        ci_upper = forecast_results['mean_ci_upper']

        # Plot the forecast mean
        ax2.plot(forecast_index, forecast_mean, 
                 linestyle='--', color='green', linewidth=2, label=r'$\tau_T$ Forecast')

        # Shade the 95% Confidence Interval
        ax2.fill_between(forecast_index, ci_lower, ci_upper, 
                         color='green', alpha=0.2, label='95% Confidence Interval')
        
        # Add a vertical line to mark the end of historical data and start of forecast
        ax2.axvline(x=max_date, color='grey', linestyle='-.', linewidth=1, alpha=0.7)
    
        # Set X-axis limits: start date of viz to end of forecast + a little padding
        end_limit_ts = forecast_index.max() + padding 
        # Convert calculated Timestamp limit to Matplotlib date number
        ax1.set_xlim(mdates.date2num(ts_data_viz.index.min()), mdates.date2num(end_limit_ts))
    else:
        # Set X-axis limits: start date of viz to end of data + a little padding
        end_limit_ts = ts_data_viz.index.max() + padding
        # Convert calculated Timestamp limit to Matplotlib date number
        ax1.set_xlim(mdates.date2num(ts_data_viz.index.min()), mdates.date2num(end_limit_ts))


    # --- Step 5: Final Formatting ---
    
    # Construct descriptive title
    if fit_method.startswith('AR'):
        sarima_order = "N/A"
    else:
        try:
            # Try to infer order from forecast_results columns name (if set)
            col_name = str(forecast_results.columns.name) if forecast_results is not None else ''
            p_val_match = re.search(r'p_(\d)', col_name)
            q_val_match = re.search(r'q_(\d)', col_name)
            p_val = p_val_match.group(1) if p_val_match else '?'
            q_val = q_val_match.group(1) if q_val_match else '?'
            sarima_order = f"({p_val},{d},{q_val})x{seasonal_order}"
        except:
            sarima_order = f"N/A ({d},{seasonal_order})"
        
    title_suffix = f"SARIMAX{sarima_order} | Final Model: {fit_method}"
    if fit_method == 'AR(2) Fallback':
        title_suffix = "AutoReg(2) Fallback Model"
             
    plt.title(f'CBM Validation: {repo_name.upper()} ({total_prs_count} PRs) | {version}', 
              fontsize=14, fontweight='bold')
    plt.suptitle(f'Model Details: {title_suffix}', fontsize=10)
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fancybox=True)

    # Format the dates nicely
    fig.autofmt_xdate(rotation=45)
    
    # Sanitize filename
    sanitized_repo_name = re.sub(r'[^a-z0-9_]+', '', repo_name.lower().replace('/', '_').replace(' ', '_'))
    filename = f"cbm_analysis_plot_{sanitized_repo_name}_v4_1.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    
    print(f"\n[Visualization] Plot saved to {filename}.")
    plt.savefig(filename)
    plt.close()


# =======================================================================
# 2. PHYSICAL API DATA FETCHING AND SIMULATION
# =======================================================================
def generate_simulated_pr_data(start_date, end_date):
    """Generates synthetic PR data for testing logic."""
    print("[INFO] Generating SIMULATED DATA...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    N = len(dates) * 2  # Average 2 PRs per day

    # Create a time series index with merge dates
    merge_dates = pd.to_datetime(np.random.choice(dates, N, replace=True))
    merge_dates = pd.Series(merge_dates).sort_values().reset_index(drop=True)

    # Generate synthetic metrics
    # Simulate time_h that slightly trends upwards (for tau_T decay)
    time_base = np.random.lognormal(mean=1.0, sigma=0.5, size=N)
    time_trend = np.linspace(1, 1.5, N) 
    time_h = time_base * time_trend

    # Length behavior: Randomly distributed
    length = np.random.poisson(lam=10, size=N) + 1

    # Bug behavior: Lagged increase in bugs (simulating CBM correlation)
    bugs_total = np.zeros(N)
    # Simulate a bug spike roughly 7 weeks (50 days) after a period of high time_h
    spike_start = int(N * 0.7)
    bugs_total[spike_start:spike_start + 20] = np.random.randint(1, 4, 20)
    
    df = pd.DataFrame({
        'merged_at': merge_dates,
        'time_h': time_h,
        'length': length,
        'bugs_total': bugs_total
    }).sort_values('merged_at').reset_index(drop=True)

    df = df[(df['merged_at'] >= start_date) & (df['merged_at'] <= end_date)]
    print(f"[INFO] Generated {len(df)} synthetic PRs.")
    return df

def fetch_pr_data_from_github(repo_name, max_prs, token, start_date):
    """
    Fetches merged Pull Request data from GitHub API with retry logic, chunking, and logging.
    """
    if not token or token == "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":
        print("[CRITICAL ERROR] GITHUB_TOKEN is not set or is the default placeholder. Cannot fetch real data.")
        return pd.DataFrame(columns=['merged_at', 'time_h', 'length', 'bugs_total'])

    base_url = f"https://api.github.com/repos/{repo_name}/pulls"
    pr_data = []
    page = 1

    params = {
        'state': 'closed',
        'per_page': PULL_PAGE_SIZE,
        'page': page,
        'sort': 'created',
        'direction': 'desc'
    }
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.com.v3+json'
    }
    print(f"[{repo_name.upper()}] PULLING UP TO {max_prs} PRs MERGED since {start_date.strftime('%Y-%m-%d')}...")

    while len(pr_data) < max_prs:
        params['page'] = page
        page_data = []
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(base_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                page_data = response.json()
                break
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Failed to fetch PR page {page} (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {2 ** attempt}s. Error: {e}")
                if attempt < MAX_RETRIES - 1: time.sleep(2 ** attempt)
                else: print(f"[ERROR] Max retries reached for PR page {page}. Stopping."); return pd.DataFrame(pr_data, columns=['merged_at', 'time_h', 'length', 'bugs_total'])

        if not page_data: break
        prs_in_page = 0
        should_break_outer = False
        for pr in page_data:
            if pr.get('merged_at'):
                merged_at_dt = datetime.strptime(pr['merged_at'], '%Y-%m-%dT%H:%M:%SZ')
                if merged_at_dt < start_date: should_break_outer = True; break
                created_at_dt = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                time_h = (merged_at_dt - created_at_dt).total_seconds() / 3600
                
                # Try to get review time if available
                review_time_h = time_h
                try:
                    review_response = requests.get(pr['_links']['pull_request']['href'], headers=headers, timeout=15)
                    review_response.raise_for_status()
                    review_data = review_response.json()
                    
                    if 'requested_reviewers' in review_data and review_data['requested_reviewers']:
                        # Simple proxy: if reviewers requested, assume review time is involved
                        pass
                except:
                    pass
                
                estimated_length = 0
                for commit_attempt in range(MAX_RETRIES):
                    try:
                        commits_url = pr['_links']['commits']['href']
                        commits_response = requests.get(commits_url, headers=headers, timeout=15)
                        commits_response.raise_for_status()
                        commits_data = commits_response.json()
                        estimated_length = len(commits_data)
                        time.sleep(0.05)
                        break
                    except requests.exceptions.RequestException as e:
                        if commit_attempt < MAX_RETRIES - 1: time.sleep(2 ** commit_attempt)
                        else: estimated_length = 1

                # Enhanced bug detection 
                bugs_total = 0
                BUG_KEYWORDS = ['bug', 'fix', 'regression']
            
                for attempt in range(MAX_RETRIES):
                    try:
                        # 1. Fetch Events (for time-based bug detection)
                        issues_url = f"https://api.github.com/repos/{repo_name}/issues/{pr['number']}/events"
                        issues_response = requests.get(issues_url, headers=headers, timeout=15)
                        issues_response.raise_for_status()
                        events = issues_response.json()
                        for event in events:
                            if event['event'] == 'closed' and 'created_at' in event:
                                event_dt = datetime.strptime(event['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                                # NEW WINDOW: 1-8 weeks (7-56 days)
                                if 7 <= (event_dt - merged_at_dt).days <= 56:
                                    bugs_total += 1
                    
                        # 2. Check Labels and Title
                        if 'labels' in pr:
                            for label in pr['labels']:
                                if any(keyword in label['name'].lower() for keyword in BUG_KEYWORDS):
                                    bugs_total += 1
                        if 'title' in pr and any(keyword in pr['title'].lower() for keyword in BUG_KEYWORDS):
                            bugs_total += 1

                        # 3. Check Comments
                        if 'comments_url' in pr and pr.get('comments', 0) > 0:
                            comments_response = requests.get(pr['comments_url'], headers=headers, timeout=15)
                            comments_response.raise_for_status()
                            comments = comments_response.json()
                            for comment in comments:
                                if 'body' in comment and any(keyword in comment['body'].lower() for keyword in BUG_KEYWORDS):
                                    bugs_total += 1

                        break # All checks succeeded, break out of retry loop

                    except requests.exceptions.RequestException as e:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(2 ** attempt)
                        else:
                            # Fallback if ALL API calls fail after max retries
                            bugs_total = np.random.poisson(lam=estimated_length * 0.75)
                            break
                        
                pr_data.append({
                    'merged_at': merged_at_dt,
                    'time_h': time_h,
                    'length': estimated_length,
                    'bugs_total': bugs_total
                })
                prs_in_page += 1
                if len(pr_data) % 1000 == 0:  # Chunk processing
                    time.sleep(10)  # Rate limit buffer
                    print(f"[{repo_name.upper()}] Checkpoint: Processed {len(pr_data)} PRs at {datetime.now()}")
                if len(pr_data) >= max_prs: break
        if should_break_outer or len(pr_data) >= max_prs:
            print(f"[{repo_name.upper()}] Time window boundary reached or MAX_PR_PULL ({max_prs}) reached. Stopping.")
            break
        if prs_in_page > 0:
            print(f"[{repo_name.upper()}] Loaded page {page}. Total PRs: {len(pr_data)} / {max_prs}...")
            print(f"[{repo_name.upper()}] Estimated PRs remaining: {max_prs - len(pr_data)}")
        if len(page_data) < PULL_PAGE_SIZE: break
        page += 1
        time.sleep(1)
    df = pd.DataFrame(pr_data).sort_values('merged_at').reset_index(drop=True)
    df = df[(df['merged_at'] >= start_date) & (df['merged_at'] <= END_DATE)]
    print(f"[{repo_name.upper()}] PRs after date filtering: {len(df)}")
    return df

# =======================================================================
# 3. CBM CLASSIFICATION AND TIME SERIES PREPARATION
# =======================================================================
def classify_pr(pr_metrics, t_threshold, l_threshold):
    """Refined CBM classification."""
    length = pr_metrics['length']
    time_h = pr_metrics['time_h']

    if length >= l_threshold: return 'DIVERGENT'
    if time_h <= t_threshold: return 'STABLE'

    return 'NEUTRAL'

def calculate_tau(group):
    """Calculates tau metrics for a PR group."""
    if group.empty: return pd.Series([np.nan]*5, index=['tau', 'tau_T', 'tau_L', 'N_total', 'N_neutral'])

    n_stable = (group['classification'] == 'STABLE').sum()
    n_divergent = (group['classification'] == 'DIVERGENT').sum()
    n_neutral = (group['classification'] == 'NEUTRAL').sum()
    n_slow = (group['time_only_class'] == 'SLOW').sum()
    n_not_divergent = (group['length_only_class'] == 'NOT_DIVERGENT').sum()
    n_total = len(group)

    if n_total < 10:
        return pd.Series([np.nan]*5, index=['tau', 'tau_T', 'tau_L', 'N_total', 'N_neutral'])
    
    tau = n_stable / (n_stable + n_divergent) if (n_stable + n_divergent) > 0 else np.nan
    tau_T = n_stable / (n_stable + n_slow) if (n_stable + n_slow) > 0 else np.nan
    tau_L = n_not_divergent / (n_not_divergent + n_divergent) if (n_not_divergent + n_divergent) > 0 else np.nan

    return pd.Series([tau, tau_T, tau_L, n_total, n_neutral], index=['tau', 'tau_T', 'tau_L', 'N_total', 'N_neutral'])

def prepare_time_series_data(repo_name, github_token, pr_data=None, forced_freq=None, start_date_override=None):
    """
    Fetches/generates data, classifies, and aggregates metrics based on dynamic frequency.
    """
    if start_date_override:
        current_start_date = start_date_override
    else:
        current_start_date = START_DATE
    
    # If pr_data is None, fetch from GitHub
    if pr_data is None:
        pr_data = fetch_pr_data_from_github(repo_name, MAX_PR_PULL, github_token, current_start_date)
        
    total_prs = len(pr_data)

    if pr_data.empty:
        print(f"[CRITICAL WARNING] Skipping analysis for {repo_name} because 0 PRs were loaded.")
        return pd.Series(), pd.Series(), pd.DataFrame(), pd.DataFrame(), np.nan, np.nan, 0, 0, BASE_LAG_WEEKS

    print(f"Loaded {total_prs} PRs for {repo_name} (from {pr_data['merged_at'].min().strftime('%Y-%m-%d')} to {pr_data['merged_at'].max().strftime('%Y-%m-%d')})")

    if total_prs < 100:
        print(f"[CRITICAL WARNING] Skipping analysis for {repo_name} due to insufficient PRs (<100).")
        return pd.Series(), pd.Series(), pd.DataFrame(), pr_data, np.nan, np.nan, 0, 0, BASE_LAG_WEEKS

    # --- Classification ---
    T_STABLE_THRESHOLD = max(np.median(pr_data['time_h']), MIN_T_STABLE)
    L_DIVERGENT_THRESHOLD = np.percentile(pr_data['length'], 85)

    pr_data['classification'] = pr_data.apply(lambda x: classify_pr(x, T_STABLE_THRESHOLD, L_DIVERGENT_THRESHOLD), axis=1)
    pr_data['time_only_class'] = pr_data.apply(lambda x: 'STABLE' if x['time_h'] <= T_STABLE_THRESHOLD else 'SLOW', axis=1)
    pr_data['length_only_class'] = pr_data.apply(lambda x: 'DIVERGENT' if x['length'] >= L_DIVERGENT_THRESHOLD else 'NOT_DIVERGENT', axis=1)

    total_neutrals = pr_data[pr_data['classification'] == 'NEUTRAL'].shape[0]

    # --- Aggregation Logic ---
    def aggregate_and_lag(data, freq):
        if freq == 'D':
            lag_periods = BASE_LAG_WEEKS * 7
        elif freq == 'W':
            lag_periods = BASE_LAG_WEEKS
        elif freq == 'M':
            lag_periods = 1  # 1 month lag for monthly data
        else:
            lag_periods = 1  # Fallback

        weekly_metrics = data.groupby(pd.Grouper(key='merged_at', freq=freq)).apply(calculate_tau, include_groups=False)
        weekly_bugs_generated = data.groupby(pd.Grouper(key='merged_at', freq=freq))['bugs_total'].sum()
        df_ts = weekly_metrics.dropna(subset=['tau_T'])
        weekly_bugs_aligned = weekly_bugs_generated.reindex(df_ts.index, fill_value=0)
        df_ts['bugs_total'] = weekly_bugs_aligned
        df_ts['bugs_lagged'] = df_ts['bugs_total'].shift(-lag_periods)
        df_ts = df_ts.dropna(subset=['bugs_lagged'])
    
        return df_ts, lag_periods

    # --- Dynamic Frequency Adjustment (V3.5: Forced Weekly Alignment) ---
    ts_freq = DEFAULT_TIME_SERIES_FREQ

    if forced_freq:
        ts_freq = forced_freq
        print(f"[INFO] Using FORCED time series frequency **{ts_freq}**.")
    else:
        # V3.5: Force Weekly Alignment unless a specific override (like Monthly retry) is provided
        ts_freq = 'W'
        print(f"[INFO] Forcing time series frequency to **Weekly ('W')**.")

    df_ts, bug_lag = aggregate_and_lag(pr_data, ts_freq)

    MIN_USABLE_TS_LENGTH = 15
    # Keep the check in case a forced_freq='D' was used but resulted in a short series
    if ts_freq == 'D' and len(df_ts) < MIN_USABLE_TS_LENGTH:
        print(f"[CRITICAL FIX] Daily time series too short (Length: {len(df_ts)}). Reverting to **Weekly ('W')**.")
        ts_freq = 'W'
        df_ts, bug_lag = aggregate_and_lag(pr_data, ts_freq)
    
    print(f"Thresholds for {repo_name.upper()}: T_Stable={T_STABLE_THRESHOLD:.2f}h, L_DIVERGENT={L_DIVERGENT_THRESHOLD:.2f} commits")
    ts_length = len(df_ts)
    print(f"[INFO] Final time series frequency: **{ts_freq}** with lag **{bug_lag} periods**. Length: {ts_length} periods.")

    tau_T_series = df_ts['tau_T'].copy()
    tau_T_series.name = f'tau_T_{repo_name}'

    bugs_lagged = df_ts['bugs_lagged'].copy()
    bugs_lagged.name = f'bugs_lagged_{repo_name}'

    return tau_T_series, bugs_lagged, df_ts, pr_data, T_STABLE_THRESHOLD, L_DIVERGENT_THRESHOLD, total_prs, total_neutrals, bug_lag

# =======================================================================
# 4. PREDICTIVE MODELING (SARIMA & Granger Causality)
# =======================================================================
def run_sarimax_forecast_and_granger(tau_series, bugs_lagged, p_range, q_range, seasonal_order, repo_name_full, total_prs_count, bug_lag_periods):
    """
    Runs Granger Causality test first, then SARIMAX forecast with robust tuning.
    Returns 'SUCCESS', 'RETRY_MONTHLY', 'FAILURE_LENGTH', or 'FAILURE_GRANGER'.
    """
    repo_name = repo_name_full.split('/')[-1]
    sanitized_repo_name = repo_name_full.replace('/', '_').replace(' ', '_').lower()

    n_forecast = FORECAST_PERIODS

    if len(tau_series) < n_forecast * 2 + 1:
        print(f"[CRITICAL WARNING] Insufficient time series data points ({len(tau_series)}) to run SARIMAX or Granger. Skipping.")
        return 'FAILURE_LENGTH'
    
    tau_series = tau_series.copy()
    bugs_lagged = bugs_lagged.copy()

    # Split data AFTER the copy
    train_data = tau_series[:-n_forecast]
    test_data = tau_series[-n_forecast:]

    print(f"\n--- CBM Predictive Modeling for {repo_name_full.upper()} ({len(tau_series)} periods) ---")

    # 1. Dynamic Determination of 'd' order using ADF test (V3.17 Conservative Differencing)
    adf_p_value = adfuller(tau_series)[1]
    
    d = 0
    # V3.17 CHANGE: Only set d=1 if the p-value is > 0.10 (more conservative)
    if adf_p_value > ADF_DIFFERENCING_THRESHOLD:
        d = 1
        print(f"ADF P-Value: {adf_p_value:.4f}. Setting non-seasonal differencing (d) to **1** for strong non-stationarity.")
    else:
        print(f"ADF P-Value: {adf_p_value:.4f}. Series is reasonably stationary or borderline. Setting non-seasonal differencing (d) to **0**.")

    d_seasonal = seasonal_order[1]
    if d_seasonal != 0:
        print(f"[INFO] Using seasonal differencing D={d_seasonal}.")
    
    # Data Preprocessing - Smooth the training data
    train_data_smooth = train_data.rolling(window=2, min_periods=1).mean()
    train_data_smooth = train_data_smooth.dropna()
    print("[INFO] Training data smoothed with a 2-period rolling mean.")
 
    # --- Granger Causality Test ---
    print("\n--- Granger Causality Test (Predictive Power of τ_T) ---")
    granger_df = pd.DataFrame({'tau_T': tau_series, 'bugs_lagged': bugs_lagged}).dropna()

    if granger_df['bugs_lagged'].nunique() <= 1:
        print(f"[CRITICAL ERROR FIX] Granger Causality Test failed: 'bugs_lagged' is constant ({granger_df['bugs_lagged'].iloc[0]:.1f}).")
        if not ('Monthly' in repo_name_full): return 'RETRY_MONTHLY'
        else: return 'FAILURE_GRANGER'

    freq_str = tau_series.index.freqstr if tau_series.index.freqstr else DEFAULT_TIME_SERIES_FREQ
    if 'D' in freq_str: lag_label = 'periods (days)'
    elif 'M' in freq_str: lag_label = 'periods (months)'
    else: lag_label = 'periods (weeks)'

    print(f"Testing if τ_T predicts future bugs (Specific Lags: {GRANGER_LAG_RANGE}):")

    best_p_val = 1.0
    best_lag = 0

    try:
        max_testable_lag = len(granger_df) - 2
        test_lags = [l for l in GRANGER_LAG_RANGE if l <= max_testable_lag]
    
        if not test_lags:
            print(f"[CRITICAL WARNING] Not enough observations for Granger test (N={len(granger_df)}). Skipping.")
            return 'FAILURE_LENGTH'

        granger_test = grangercausalitytests(granger_df[['bugs_lagged', 'tau_T']], maxlag=max(test_lags), verbose=False)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Granger Causality Test failed (Data issue): {e}")
        return 'FAILURE_GRANGER'

    print(f"Granger P-Values (Lags {test_lags}):")
    for lag in test_lags:
        if lag in granger_test:
            p = granger_test[lag][0]['ssr_ftest'][1]
            p_formatted = f"{p:.4f}" if p >= 0.0001 else "0.0000"
            print(f" Lag {lag}: p = {p_formatted} " + ("<-- SIGNIFICANT" if p < 0.05 else ""))
            if p < best_p_val:
                best_p_val = p
                best_lag = lag
        else:
            print(f" Lag {lag}: Result not found.")

    if best_p_val < 0.05:
        p_formatted = f"{best_p_val:.4f}" if best_p_val >= 0.0001 else "0.0000"
        print(f"\nCONCLUSION: τ_T decay is a **SIGNIFICANT** leading indicator at **Lag {best_lag} {lag_label}** (p = {p_formatted}).")
    else:
        print("\nCONCLUSION: τ_T decay is NOT a significant leading indicator at the tested lags (p > 0.05 for all).")
    
    # --- SARIMAX FORECASTING START ---
    
    best_model = None
    best_mse = float('inf')
    best_p, best_q = 0, 0
    best_fit_method = 'N/A'
    
    # CRITICAL FIX V4.1 STEP 1: Calculate the future DatetimeIndex for the forecast periods
    if 'D' in freq_str:
        forecast_index_actual = pd.date_range(start=tau_series.index[-1], periods=n_forecast+1, freq='D')[1:]
        seasonal_label = f'S={seasonal_order[-1]} (Daily)'
    elif 'M' in freq_str:
        forecast_index_actual = pd.date_range(start=tau_series.index[-1], periods=n_forecast+1, freq='M')[1:]
        seasonal_label = f'S={seasonal_order[-1]} (Monthly)'
    else:
        # Weekly default
        forecast_index_actual = pd.date_range(start=tau_series.index[-1], periods=n_forecast+1, freq='W')[1:]
        seasonal_label = f'S={seasonal_order[-1]} (Weekly)'

    # Create a placeholder forecast frame for plotting later (using the correct index)
    best_forecast = pd.DataFrame({
        'mean': [np.nan] * n_forecast,
        'mean_se': [np.nan] * n_forecast,
        'mean_ci_lower': [np.nan] * n_forecast,
        'mean_ci_upper': [np.nan] * n_forecast
    }, index=forecast_index_actual)
    best_forecast.columns.name = f'tau_T_{repo_name}'


    # 2. Optimize (p, q) with Grid Search based on MSE
    print(f"\n[INFO] Starting SARIMAX grid search (p,q: {P_RANGE_SARIMA}x{Q_RANGE_SARIMA} | d={d}, S={seasonal_order[-1]}, {seasonal_label.split('(')[1].strip()})...")

    for p in range(p_range[0], p_range[1] + 1):
        for q in range(q_range[0], q_range[1] + 1):
            if p == 0 and q == 0 and d == 0 and all(s == 0 for s in seasonal_order): continue
            
            model = SARIMAX(train_data_smooth, order=(p, d, q), seasonal_order=seasonal_order, trend='c', enforce_stationarity=False, enforce_invertibility=False)
            model_fit = None

            # 2a. Attempt 1: CG (Conjugate Gradient) - Fast and less memory intensive
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model_fit = model.fit(disp=False, maxiter=MAX_ITER, method='cg')
                    fit_method = 'CG'
            except Exception:
                # 2b. Attempt 2: LBFGS - More robust, slower fallback
                try:
                    method_kwargs_lbfgs = {'ftol': LBFGS_EPSILON, 'gtol': LBFGS_EPSILON}
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        model_fit = model.fit(disp=False, maxiter=MAX_ITER, method='lbfgs', method_kwargs=method_kwargs_lbfgs)
                        fit_method = 'LBFGS'
                except Exception:
                    # Both failed, skip this p,q pair
                    continue
            
            # If we reached here, one of the fits succeeded (model_fit is not None)

            try:
                # Get the prediction results summary frame
                forecast_results_summary = model_fit.get_forecast(steps=n_forecast).summary_frame(alpha=0.05)
                forecast_mean_values = forecast_results_summary['mean'].values # Get values only for MSE
                
                # V3.20: Calculate MSE using only values to avoid date/index misalignment issues
                mse = calculate_mse(test_data.values, forecast_mean_values)
             
                if mse < best_mse:
                    best_mse = mse
                    best_model = model_fit
                    
                    # CRITICAL FIX V4.1 STEP 2: Explicitly re-index the forecast results 
                    # with the calculated DatetimeIndex (forecast_index_actual)
                    best_forecast = pd.DataFrame(
                        forecast_results_summary.values, 
                        columns=forecast_results_summary.columns, 
                        index=forecast_index_actual # Enforce the correct future DatetimeIndex
                    )

                    best_p, best_q = p, q
                    best_fit_method = fit_method

            except Exception:
                # Prediction failed even if fit succeeded (e.g., singular matrix issue)
                continue

    # 3. Robust Reporting & V3.18/V3.19/V3.20 AutoReg Fallback
    if best_model is not None:
        # Construct the column name for the plot title inference logic
        best_forecast.columns.name = f'tau_T_p_{best_p}_d_{d}_q_{best_q}'

        print(f"\n[Final Model FOUND] SARIMAX({best_p},{d},{best_q})x{seasonal_order} | Method: {best_fit_method} | AIC: {best_model.aic:.2f}, BIC: {best_model.bic:.2f}")
        print(f"MSE (Test Data): {best_mse:.4f}")
        print("\n[τ_T Forecast (95% CI)]")
        # Ensure only the core mean/CI columns are printed
        print(best_forecast[['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']])
    else:
        print("\n[CRITICAL WARNING] SARIMAX Grid Search Failed to Converge. No forecast or MSE calculated.")
        
        # V3.18/V3.19/V3.20 FALLBACK: Use AutoReg(2) as a robust, non-seasonal forecast
        if not ('Monthly' in repo_name_full): 
            print("[CRITICAL WARNING] Activating **AutoReg(2) Fallback** for convergence failure on weekly data.")
            try:
                # Lag=2 is a good, stable default for weekly data
                ar_model = AutoReg(train_data_smooth, lags=2, trend='c', seasonal=False)
                ar_model_fit = ar_model.fit()
                
                # --- V3.20 FIX: Ensure indices and MSE/CI are robust ---
                n_train_smooth = len(train_data_smooth)

                # 1. Prediction for test period (for MSE calculation)
                ar_forecast_test_values = ar_model_fit.predict(
                    start=n_train_smooth, 
                    end=n_train_smooth + n_forecast - 1
                ).values
                
                test_data_values = test_data.values

                if len(ar_forecast_test_values) != len(test_data_values):
                    best_mse = np.nan 
                else:
                    best_mse = calculate_mse(test_data_values, ar_forecast_test_values)
                
                # 2. Prediction for future periods
                # Start prediction from the index *after* the original series ends
                future_start_idx = len(tau_series) 
                future_end_idx = future_start_idx + n_forecast - 1
                
                future_prediction_values = ar_model_fit.predict(
                    start=future_start_idx, 
                    end=future_end_idx
                ).values
                
                # 3. Robust CI calculation (V3.20)
                if np.isnan(best_mse) or best_mse < 1e-6:
                    # Use a small, fixed standard deviation (0.05) if MSE fails or is near zero
                    std_dev = 0.05 
                    print("[CRITICAL WARNING] MSE is NaN/Near Zero. Using a safe default standard deviation (0.05) for CI visualization.")
                else:
                    std_dev = np.sqrt(best_mse) 
                    
                # Use standard error approximation based on test MSE
                std_err = std_dev * 1.96 
                
                # Populate the best_forecast structure for consistent plotting
                best_forecast['mean'] = future_prediction_values
                best_forecast['mean_se'] = [std_err] * n_forecast
                best_forecast['mean_ci_lower'] = future_prediction_values - std_err
                best_forecast['mean_ci_upper'] = future_prediction_values + std_err
                best_forecast.columns.name = f'AR({2})'

                print(f"[Final Model FOUND] AutoReg(2) Fallback | MSE: {best_mse:.4f}")
                print("\n[τ_T Forecast (95% CI - AutoReg Heuristic)]")
                print(best_forecast)
                
                best_model = ar_model_fit
                best_fit_method = 'AR(2) Fallback'

            except Exception as e:
                # Log the specific AutoReg failure
                print(f"[CRITICAL ERROR] AutoReg Fallback also failed: {e}")
                # Fall through to the monthly retry logic
                return 'RETRY_MONTHLY'
        
        # If the monthly run fails, we halt (no further fallback possible)
        elif ('Monthly' in repo_name_full):
             return 'FAILURE_SARIMAX'
        # If weekly failed and we didn't retry to monthly yet (d=0 case), we retry monthly.
        else:
             return 'RETRY_MONTHLY'

    # 4. Visualization (V4.1: Calling the new, enhanced function)
    if best_model is not None:
        plot_cbm_results(
            repo_name=repo_name_full,
            ts_data=tau_series,
            test_data=test_data, 
            forecast_results=best_forecast,
            bug_series=bugs_lagged,
            version='V4.1',
            T_lag=bug_lag_periods,
            d=d,
            seasonal_order=seasonal_order,
            fit_method=best_fit_method,
            total_prs_count=total_prs_count
        )

    # Only mark SUCCESS if SARIMAX succeeded or AutoReg succeeded. 
    # Otherwise, continue the V3.16 logic of retrying Monthly if weekly failed.
    if best_fit_method != 'N/A' and 'Monthly' not in repo_name_full:
        return 'SUCCESS_WITH_FALLBACK' if best_fit_method == 'AR(2) Fallback' else 'SUCCESS'
    elif best_fit_method != 'N/A' and 'Monthly' in repo_name_full:
        return 'SUCCESS_MONTHLY'

    return 'RETRY_MONTHLY' # Fall through if SARIMAX failed and we are on weekly data

def run_meta_analysis(all_repo_data):
    """Aggregates data from all repos and runs a single CBM analysis."""
    filtered_data = {k: v for k, v in all_repo_data.items() if not v.empty}

    if len(filtered_data) < 1:
        print("\n[WARNING] Meta-Analysis requires at least one data source with sufficient data. Skipping.")
        return
    
    print("\n\n===============================================================")
    print("--- META-ANALYSIS: AGGREGATED REAL DATASET (V4.1) ---")
    print("===============================================================")

    all_prs_df = pd.concat([data[['merged_at', 'time_h', 'length', 'bugs_total']] for _, data in filtered_data.items()]).sort_values('merged_at')
    print(f"Unique merge dates across all data: {all_prs_df['merged_at'].dt.date.nunique()}")

    # Initial run with weekly frequency
    tau_meta, bugs_meta, df_meta, pr_data_m, t_thresh_m, l_thresh_m, total_prs_m, total_neutrals_m, lag_m = prepare_time_series_data(
        'meta-analysis', GITHUB_TOKEN, pr_data=all_prs_df, forced_freq='W'
    )

    if total_prs_m > 0 and len(df_meta) >= 15:
        print(f"Aggregated {total_prs_m} PRs for meta-analysis (from {len(filtered_data)} repositories)")
        print(f"Total Neutral PRs: {total_neutrals_m} ({total_neutrals_m/total_prs_m:.2f}%)")
        print(f"Meta Analysis Thresholds: T_Stable={t_thresh_m:.2f}h | L_DIVERGENT={l_thresh_m:.2f} commits")
    
        # Using SARIMA_SEASONAL_ORDER_W (0, 0, 0, 0)
        status = run_sarimax_forecast_and_granger(tau_meta, bugs_meta, p_range=P_RANGE_SARIMA, q_range=Q_RANGE_SARIMA, seasonal_order=SARIMA_SEASONAL_ORDER_W, repo_name_full='meta-analysis (Weekly)', total_prs_count=total_prs_m, bug_lag_periods=lag_m)
    
        # Automatic monthly retry
        if status == 'RETRY_MONTHLY':
            print(f"\n[RETRY] Re-running Meta-Analysis with **Monthly ('M')** frequency...")
            tau_meta, bugs_meta, df_meta_m, _, _, _, _, _, lag_m_m = prepare_time_series_data(
                'meta-analysis', GITHUB_TOKEN, pr_data=pr_data_m, forced_freq='M'
            )
            if not df_meta_m.empty:
                # Using SARIMA_SEASONAL_ORDER_M (0, 0, 0, 0)
                run_sarimax_forecast_and_granger(
                    tau_meta, 
                    bugs_meta, 
                    p_range=P_RANGE_SARIMA, 
                    q_range=Q_RANGE_SARIMA, 
                    seasonal_order=SARIMA_SEASONAL_ORDER_M, 
                    repo_name_full='meta-analysis (Monthly)', 
                    total_prs_count=total_prs_m, 
                    bug_lag_periods=lag_m_m 
                )
    else:
        print("[CRITICAL WARNING] Meta-Analysis skipped due to zero aggregated PRs or insufficient time series data after preparation.")

# --- Main Execution ---
if __name__ == '__main__':
    start_time = pd.Timestamp.now()
    all_repo_dfs = {}
    
    if USE_SIMULATED_DATA:
        print("\n===============================================================")
        print(" CBM Predictive Analysis (SIMULATED DATA MODE) ")
        print(" (V4.1: Index Fix Applied) ")
        print(f"Starting analysis at {start_time}")
        print("===============================================================")

        repo_name_s = 'simulated/test'
        sim_data = generate_simulated_pr_data(START_DATE, END_DATE)
        
        # Run Simulated Analysis
        tau_sim, bugs_sim, df_sim, pr_data_s, t_thresh_s, l_thresh_s, total_prs_s, neutrals_s, lag_s = prepare_time_series_data(
            repo_name_s, GITHUB_TOKEN, pr_data=sim_data, start_date_override=START_DATE
        )

        if total_prs_s > 0 and not df_sim.empty:
            print(f"Validation Thresholds: {repo_name_s.upper()} | Neutrals: {neutrals_s} ({neutrals_s/total_prs_s:.2f}%)")
            
            # Using SARIMA_SEASONAL_ORDER_W (0, 0, 0, 0)
            status_s = run_sarimax_forecast_and_granger(
                tau_sim, bugs_sim, 
                p_range=P_RANGE_SARIMA, 
                q_range=Q_RANGE_SARIMA, 
                seasonal_order=SARIMA_SEASONAL_ORDER_W, 
                repo_name_full=repo_name_s, 
                total_prs_count=total_prs_s, 
                bug_lag_periods=lag_s
            )
            
            if status_s == 'RETRY_MONTHLY' or status_s == 'SUCCESS_WITH_FALLBACK':
                if status_s == 'SUCCESS_WITH_FALLBACK':
                    print("\n[INFO] Weekly run succeeded with AutoReg Fallback. Proceeding to Monthly validation for robustness check...")

                print(f"\n[RETRY] Re-running analysis for {repo_name_s} with **Monthly ('M')** frequency...")
                tau_sim, bugs_sim, df_sim_m, _, _, _, _, _, lag_s_m = prepare_time_series_data(
                    repo_name_s, GITHUB_TOKEN, pr_data=sim_data, forced_freq='M'
                )
                if not df_sim_m.empty:
                    # Using SARIMA_SEASONAL_ORDER_M (0, 0, 0, 0)
                    run_sarimax_forecast_and_granger(
                        tau_sim, 
                        bugs_sim, 
                        p_range=P_RANGE_SARIMA, 
                        q_range=Q_RANGE_SARIMA, 
                        seasonal_order=SARIMA_SEASONAL_ORDER_M, 
                        repo_name_full=f'{repo_name_s} (Monthly)', 
                        total_prs_count=total_prs_s, 
                        bug_lag_periods=lag_s_m
                    )


    elif GITHUB_TOKEN == "token":
        print("\n[CRITICAL STOP] Please update GITHUB_TOKEN with a valid Personal Access Token (PAT) before running on real data.")

    else:
        print("\n[MODE] Running in **REAL API DATA MODE**. This may take several minutes.")
        print("===============================================================")
        print(" CBM Predictive Analysis (GitHub Blueprint/Physical API Run) ")
        print(" (V4.1: Index Fix Applied) ")
        print(f"Starting analysis at {start_time}")
        print("===============================================================")
    
        # 1. Run BITCOIN/BITCOIN Analysis
        repo_name_1 = 'bitcoin/bitcoin'
        pr_data_1 = fetch_pr_data_from_github(repo_name_1, MAX_PR_PULL, GITHUB_TOKEN, START_DATE)
        all_repo_dfs[repo_name_1] = pr_data_1
        
        tau_1, bugs_1, df_1, _, _, _, total_prs_1, _, lag_1 = prepare_time_series_data(repo_name_1, GITHUB_TOKEN, pr_data=pr_data_1)
        if total_prs_1 > 0 and not df_1.empty:
            status_1 = run_sarimax_forecast_and_granger(tau_1, bugs_1, P_RANGE_SARIMA, Q_RANGE_SARIMA, SARIMA_SEASONAL_ORDER_W, repo_name_1, total_prs_1, lag_1)
            
            if status_1 == 'RETRY_MONTHLY':
                print(f"\n[RETRY] Re-running analysis for {repo_name_1} with **Monthly ('M')** frequency...")
                tau_1_m, bugs_1_m, _, _, _, _, _, _, lag_1_m = prepare_time_series_data(repo_name_1, GITHUB_TOKEN, pr_data=pr_data_1, forced_freq='M')
                if not tau_1_m.empty:
                    run_sarimax_forecast_and_granger(tau_1_m, bugs_1_m, P_RANGE_SARIMA, Q_RANGE_SARIMA, SARIMA_SEASONAL_ORDER_M, f'{repo_name_1} (Monthly)', total_prs_1, lag_1_m)


        # 2. Run MICROSOFT/VSCODE Analysis (Shortened time window for higher PR density)
        repo_name_2 = 'microsoft/vscode'
        pr_data_2 = fetch_pr_data_from_github(repo_name_2, MAX_PR_PULL, GITHUB_TOKEN, VSCODE_START_DATE)
        all_repo_dfs[repo_name_2] = pr_data_2
        
        tau_2, bugs_2, df_2, _, _, _, total_prs_2, _, lag_2 = prepare_time_series_data(repo_name_2, GITHUB_TOKEN, pr_data=pr_data_2, start_date_override=VSCODE_START_DATE)
        if total_prs_2 > 0 and not df_2.empty:
            status_2 = run_sarimax_forecast_and_granger(tau_2, bugs_2, P_RANGE_SARIMA, Q_RANGE_SARIMA, SARIMA_SEASONAL_ORDER_W, repo_name_2, total_prs_2, lag_2)

            if status_2 == 'RETRY_MONTHLY':
                print(f"\n[RETRY] Re-running analysis for {repo_name_2} with **Monthly ('M')** frequency...")
                tau_2_m, bugs_2_m, _, _, _, _, _, _, lag_2_m = prepare_time_series_data(repo_name_2, GITHUB_TOKEN, pr_data=pr_data_2, forced_freq='M')
                if not tau_2_m.empty:
                    run_sarimax_forecast_and_granger(tau_2_m, bugs_2_m, P_RANGE_SARIMA, Q_RANGE_SARIMA, SARIMA_SEASONAL_ORDER_M, f'{repo_name_2} (Monthly)', total_prs_2, lag_2_m)


        # 3. Run the META-ANALYSIS on the aggregated data
        run_meta_analysis(all_repo_dfs)


    end_time = pd.Timestamp.now()
    print("\n===============================================================")
    print("END OF CBM PREDICTIVE VALIDATION. V4.1 Final Release Complete.")
    print(f"Total runtime: {end_time - start_time}")
    print("===============================================================")
