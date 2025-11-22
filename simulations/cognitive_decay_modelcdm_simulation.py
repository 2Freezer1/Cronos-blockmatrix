Here my code:

# V4.1: FINAL CONSOLIDATED VERSION - ADAPTED FOR WIKIPEDIA CDM DATA
# FIX: Corrected SARIMAX prediction indices (start_idx and end_idx) to be relative 
# to the training data length (tau_train) instead of the full time series (tau_ts).

# --- Library Imports ---
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.ar_model import AutoReg 
import warnings
from datetime import datetime, timedelta

import matplotlib
# Set backend to Agg for non-interactive environments (prevents GUI errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

# Suppress the harmless future warnings from statsmodels
warnings.filterwarnings("ignore")

# --- Configuration Constants (Adapted from V4.1) ---
P_RANGE_SARIMA = range(0, 2)
Q_RANGE_SARIMA = range(0, 2)
# Using simple ARMA(1,1) for this demo. Set to (0,0,0,0) for no seasonality.
SARIMA_SEASONAL_ORDER_W = (0, 0, 0, 0) 
FORECAST_STEPS = 3
GRANGER_TEST_MAXLAG = 3 # Increased maxlag for more data points
ALPHA_THRESHOLD = 0.05
MOCK_WEEKS = 20 # Significantly increase mock data size

# ==============================================================================
# 1. WIKIPEDIA CDM DATA LOADING AND FEATURE EXTRACTION
#    (Modified for extensive mock data generation)
# ==============================================================================

def load_search_logs():
    """
    Loads MOCK search logs for 20 weeks to enable statistical tests.
    In a real-world scenario, this would be replaced with actual log fetching.
    """
    print(f"Generating {MOCK_WEEKS} weeks of mock search log data...")
    
    start_date = datetime.now() - timedelta(weeks=MOCK_WEEKS)
    all_data = []
    
    for i in range(MOCK_WEEKS):
        current_date = start_date + timedelta(weeks=i)
        
        # Simulate 5-15 sessions per week
        num_sessions = np.random.randint(5, 16)
        
        for j in range(num_sessions):
            session_id = f"{i}_{j}"
            # Simulate 2-5 interactions per session
            num_interactions = np.random.randint(2, 6)
            
            query = f"Search Query {i+j}"
            
            for k in range(num_interactions):
                # Generate timestamp within a few minutes of the week's start
                ts = current_date + timedelta(minutes=np.random.randint(0, 60), seconds=np.random.randint(0, 60))
                
                # Introduce clicks and code searches randomly
                is_clicked = 1 if np.random.rand() < 0.3 else 0
                query_with_code = query if np.random.rand() < 0.2 else f"Python {query}"
                
                all_data.append({
                    'SessionID': session_id,
                    'Query': query_with_code if 'Python' in query_with_code else query,
                    'PageURL': f'/result_{i}_{j}_{k}',
                    'Rank': k + 1,
                    'IsClicked': is_clicked,
                    'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S')
                })

    df = pd.DataFrame(all_data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def extract_sdm_features(df):
    """
    Calculates the Search Decay Model (SDM) features, including the
    cognitive collapse risk ($\tau_T$) and aggregates them weekly.
    (Feature calculation logic remains the same)
    """
    print("\n[STEP 1/3] SDM Feature Extraction: Cognitive Load $\\rightarrow$ Collapse Risk")

    df = df.sort_values(['SessionID', 'Timestamp'])

    # === 1. QUERY COMPLEXITY (C) ===
    df['C_query_length'] = df['Query'].apply(lambda x: len(x.split()))
    
    def calculate_entropy(query):
        words = query.lower().split()
        if not words: return 0.0
        word_counts = pd.Series(words).value_counts()
        probabilities = word_counts / len(words)
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
        
    df['C_query_entropy'] = df['Query'].apply(calculate_entropy)
    df['C_has_code'] = df['Query'].str.contains(r'python|javascript|sql', case=False).astype(int)

    # === 2. CLICK SPEED (τ) ===
    df['session_start_time'] = df.groupby('SessionID')['Timestamp'].transform('min')
    first_click = df[df['IsClicked'] == 1].groupby('SessionID')['Timestamp'].min()
    df = df.merge(first_click.rename('first_click_time'), on='SessionID', how='left')
    df['τ_to_first_click'] = (df['first_click_time'] - df['session_start_time']).dt.total_seconds()
    
    # === 3. LATENT FAILURE (τ_T) INDICATORS ===
    session_durations = df.groupby('SessionID')['Timestamp'].agg(['min', 'max'])
    session_durations['duration'] = (session_durations['max'] - session_durations['min']).dt.total_seconds()
    
    # Re-calculate abandonment more robustly after grouping
    max_click_per_session = df.groupby('SessionID')['IsClicked'].max().fillna(0)
    session_durations['abandoned'] = ((session_durations['duration'] < 30) & (max_click_per_session == 0)).astype(int)

    df = df.merge(session_durations[['abandoned']], on='SessionID', how='left')

    df['pogo'] = 0
    for session_id, group in df[df['IsClicked'] == 1].groupby('SessionID'):
        clicks = group.sort_values('Timestamp')
        if len(clicks) > 1 and (clicks['Timestamp'].diff().dt.total_seconds() < 10).any():
            df.loc[df['SessionID'] == session_id, 'pogo'] = 1
                
    # $\tau_T$ = cognitive collapse risk (Weighted average)
    df['τ_T'] = (
        0.4 * df['C_query_length'].clip(0, 10)/10 +
        0.3 * (1 / (df['τ_to_first_click'].fillna(60).clip(1, 60) / 60)) +
        0.2 * df['pogo'] +
        0.1 * df['abandoned']
    )

    # === 4. WEEKLY AGGREGATION ===
    df['week'] = df['Timestamp'].dt.to_period('W')
    
    weekly = df.groupby('week').agg(
        τ_T_mean=('τ_T', 'mean'),
        click_count=('IsClicked', 'sum'),
        session_count=('SessionID', 'nunique')
    ).rename(columns={'click_count': 'bug_count_T3'}) # Bug proxy: total clicks in the week

    return weekly

# ==============================================================================
# 2. CBM V4.1 TIME SERIES PREPARATION AND SARIMAX/GRANGER FUNCTIONS
# ==============================================================================

def prepare_time_series_data(repo_name, weekly_series, forced_freq='W'):
    """
    Prepares the weekly $\tau_T$ and 'bug count' time series for analysis.
    """
    print(f"\n[STEP 2/3] Preparing Time Series for {repo_name} (Freq: {forced_freq})...")
    
    series_df = weekly_series.copy()
    series_df.index = series_df.index.to_timestamp(how='end')

    # Ensure continuous time series (Critical for SARIMAX)
    start_date = series_df.index.min()
    end_date = series_df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=forced_freq)
    
    # Reindex the data, forward-fill $\tau_T$ and fill bug/session counts with 0
    series_df = series_df.reindex(full_date_range)
    # Fill NaN $\tau_T$ means using the last valid mean
    series_df['τ_T_mean'] = series_df['τ_T_mean'].fillna(method='ffill').fillna(series_df['τ_T_mean'].mean())
    series_df[['bug_count_T3', 'session_count']] = series_df[['bug_count_T3', 'session_count']].fillna(0)
    
    tau_ts = series_df['τ_T_mean']
    bugs_ts = series_df['bug_count_T3']
    
    print(f"Time Series Prepared. Total Data Points: {len(series_df)}")
    
    total_data_points = len(series_df)
    lag_weeks = 1 # Mocked lag
    
    # Returns the necessary components (many are kept None for compatibility with original V4.1)
    return tau_ts, bugs_ts, series_df, None, None, None, total_data_points, None, lag_weeks

def plot_forecast(tau_ts, sarima_forecast, ar_forecast, repo_name, target_variable='τ_T_mean'):
    """
    Plots the historical $\tau_T$ and the SARIMAX/AR forecasts.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(tau_ts.index, tau_ts.values, label='Historical $\\tau_T$ (Collapse Risk)', color='darkblue', marker='o', linestyle='-')
    
    # Plot SARIMAX forecast
    if sarima_forecast is not None:
        ax.plot(sarima_forecast.index, sarima_forecast.values, label='SARIMAX Forecast', color='red', linestyle='--', marker='x')

    # Plot AR forecast (if used as fallback)
    if ar_forecast is not None:
        ax.plot(ar_forecast.index, ar_forecast.values, label='AR Fallback Forecast', color='orange', linestyle=':', marker='s')

    # Formatting
    ax.set_title(f'Weekly $\\tau_T$ (Collapse Risk) and Forecast for {repo_name}')
    ax.set_ylabel('Avg. $\\tau_T$ Index')
    ax.set_xlabel('Date (End of Week)')
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Date formatting for x-axis
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{repo_name}_tau_T_forecast.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"\n[PLOT GENERATED]: Plot saved to {plot_filename}")

def run_sarimax_forecast_and_granger(tau_ts, bugs_ts, p_range, q_range, seasonal_order, repo_name, total_data_points, lag_weeks, target_variable='τ_T_mean'):
    """
    Performs SARIMAX forecasting, Granger Causality test, and prepares plot data.
    """
    print(f"\n--- Analysis for {repo_name} (Data Points: {total_data_points}) ---")
    
    sarima_forecast = None
    ar_forecast = None
    tau_train = tau_ts.iloc[:-FORECAST_STEPS] 
    
    # --- 1. Granger Causality Test ($\tau_T \rightarrow$ Clicks) ---
    granger_data = pd.DataFrame({'Bugs': bugs_ts.values, 'Tau': tau_ts.values})
    
    if len(granger_data) > GRANGER_TEST_MAXLAG:
        try:
            print(f"\n[GRANGER CAUSALITY TEST] Max Lag: {GRANGER_TEST_MAXLAG}")
            gc_results = grangercausalitytests(granger_data[['Bugs', 'Tau']], maxlag=[GRANGER_TEST_MAXLAG], verbose=False)
            p_value = gc_results[GRANGER_TEST_MAXLAG][0]['ssr_ftest'][1]
            
            if p_value < ALPHA_THRESHOLD:
                print(f"\\textbf{{[CAUSALITY FOUND]}}: $\\tau_T \\rightarrow$ Clicks ($p={p_value:.4f}$). Reject $H_0$.")
            else:
                print(f"[NO CAUSALITY]: $\\tau_T \\nrightarrow$ Clicks ($p={p_value:.4f}$). Cannot reject $H_0$.")

        except Exception as e:
            print(f"[GRANGER ERROR]: Could not run test due to error: {e}")
    else:
        print(f"[GRANGER SKIPPED]: Need more than {GRANGER_TEST_MAXLAG} data points.")
    
    # --- 2. SARIMAX/AutoReg Forecasting ---
    if len(tau_train) < 3:
        print("[FORECAST SKIPPED]: Too few data points for training.")
        plot_forecast(tau_ts, None, None, repo_name, target_variable)
        return 'INSUFFICIENT_DATA'
        
    try:
        print("\n[SARIMAX FORECAST]")
        
        # Simplified SARIMAX fit (p=1, d=0, q=1)
        sarima_model = SARIMAX(
            tau_train, 
            order=(1, 0, 1), 
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        # --- FIX APPLIED HERE ---
        # Calculate prediction indices relative to the training set (tau_train).
        # start_idx is the index immediately following the training set.
        start_idx = len(tau_train)
        # end_idx is the index of the last forecasted step.
        end_idx = len(tau_train) + FORECAST_STEPS - 1
        
        # Predict the next N steps
        sarima_forecast = sarima_model.predict(start=start_idx, end=end_idx)
        # ------------------------

        # Set the forecast index correctly based on the original series index
        forecast_index = pd.date_range(
            start=tau_ts.index[-1] + timedelta(days=7), 
            periods=FORECAST_STEPS, 
            freq='W-SUN'
        )
        sarima_forecast.index = forecast_index

        print(f"\nSARIMAX Forecast ({target_variable}, {FORECAST_STEPS} Periods):\n{sarima_forecast.to_string()}")
        plot_forecast(tau_ts, sarima_forecast, None, repo_name, target_variable)
        return 'SUCCESS'

    except Exception as sarimax_e:
        print(f"[SARIMAX FAILED]: {sarimax_e}. Falling back to AutoReg (AR) Model.")
        
        # --- AutoReg Fallback ---
        try:
            # Use AutoReg as a simpler model for robustness
            ar_model = AutoReg(tau_train, lags=1).fit()
            
            # Predict
            ar_forecast_raw = ar_model.predict(start=len(tau_train), end=len(tau_train) + FORECAST_STEPS - 1, dynamic=False)
            
            # Set the forecast index correctly
            ar_forecast_raw.index = pd.date_range(
                start=tau_ts.index[-1] + timedelta(days=7), 
                periods=FORECAST_STEPS, 
                freq='W-SUN'
            )
            ar_forecast = ar_forecast_raw

            print(f"\n[AR FALLBACK SUCCESS]: Forecast:\n{ar_forecast.to_string()}")
            plot_forecast(tau_ts, None, ar_forecast, repo_name, target_variable)
            return 'AR_FALLBACK_SUCCESS'

        except Exception as ar_e:
            print(f"[AUTOREG FAILED]: {ar_e}. Cannot forecast.")
            plot_forecast(tau_ts, None, None, repo_name, target_variable)
            return 'FORECAST_FAILED'


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Load Data and Extract SDM Features ---
    raw_logs_df = load_search_logs()
    weekly_series = extract_sdm_features(raw_logs_df)

    # --- 2. Prepare Time Series and Run Analysis ---
    
    ANALYSIS_NAME = 'Wikipedia_CDM_Search_Logs'
    
    tau_ts, bugs_ts, ts_df, _, _, _, total_data_points, _, lag_weeks = prepare_time_series_data(
        ANALYSIS_NAME, 
        weekly_series, 
        forced_freq='W' # Weekly aggregation
    )

    if total_data_points > 0 and not ts_df.empty:
        run_sarimax_forecast_and_granger(
            tau_ts, 
            bugs_ts, 
            P_RANGE_SARIMA, 
            Q_RANGE_SARIMA, 
            SARIMA_SEASONAL_ORDER_W, 
            ANALYSIS_NAME, 
            total_data_points, 
            lag_weeks
        )
    else:
        print("\n[STATUS] Data preparation failed or resulted in zero data points.")

    print("\n\n--- Wikipedia CDM Analysis Complete ---")
    print(f"Check the file '{ANALYSIS_NAME}_tau_T_forecast.png' for the generated plot.")

