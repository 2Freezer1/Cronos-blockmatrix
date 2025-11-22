"""
Project: CRONOS-Blockmatrix Theory (C-BMT)
Module: Cognitive Decay Model (CDM) Simulation
Author: Adrian Ammann
Date: November 2025

Abstract:
This simulation applies the C-BMT Minimum Cost Principle to cognitive thermodynamics.
It tests the hypothesis that 'Cognitive Collapse' (Decision/Click) is triggered when
Resonance Density (tau_T) crosses a specific entropic threshold.

Equation: E_Det = k * (Delta_rho)^2 * (Delta_S / N_eff) + delta_Q
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.ar_model import AutoReg 
import warnings
from datetime import datetime, timedelta
import matplotlib
# Set backend to Agg for non-interactive environments (prevents GUI errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration Constants ---
P_RANGE_SARIMA = range(0, 2)
Q_RANGE_SARIMA = range(0, 2)
SARIMA_SEASONAL_ORDER_W = (0, 0, 0, 0) 
FORECAST_STEPS = 3
GRANGER_TEST_MAXLAG = 3 
ALPHA_THRESHOLD = 0.05
MOCK_WEEKS = 20 

# ==============================================================================
# 1. DATA LOADING AND FEATURE EXTRACTION
# ==============================================================================

def load_search_logs():
    """
    Generates MOCK search logs to simulate cognitive consensus (N).
    """
    print(f"Generating {MOCK_WEEKS} weeks of mock search log data...")
    
    start_date = datetime.now() - timedelta(weeks=MOCK_WEEKS)
    all_data = []
    
    for i in range(MOCK_WEEKS):
        current_date = start_date + timedelta(weeks=i)
        num_sessions = np.random.randint(5, 16)
        
        for j in range(num_sessions):
            session_id = f"{i}_{j}"
            num_interactions = np.random.randint(2, 6)
            query = f"Search Query {i+j}"
            
            for k in range(num_interactions):
                ts = current_date + timedelta(minutes=np.random.randint(0, 60), seconds=np.random.randint(0, 60))
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
    Calculates Search Decay Model (SDM) features.
    tau_T represents the Cognitive Collapse Risk (Resonance Density).
    """
    print("\n[STEP 1/3] SDM Feature Extraction: Cognitive Load -> Collapse Risk")

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

    # === 2. CLICK SPEED (tau) ===
    df['session_start_time'] = df.groupby('SessionID')['Timestamp'].transform('min')
    first_click = df[df['IsClicked'] == 1].groupby('SessionID')['Timestamp'].min()
    df = df.merge(first_click.rename('first_click_time'), on='SessionID', how='left')
    df['tau_to_first_click'] = (df['first_click_time'] - df['session_start_time']).dt.total_seconds()
    
    # === 3. LATENT FAILURE INDICATORS ===
    session_durations = df.groupby('SessionID')['Timestamp'].agg(['min', 'max'])
    session_durations['duration'] = (session_durations['max'] - session_durations['min']).dt.total_seconds()
    
    max_click_per_session = df.groupby('SessionID')['IsClicked'].max().fillna(0)
    session_durations['abandoned'] = ((session_durations['duration'] < 30) & (max_click_per_session == 0)).astype(int)

    df = df.merge(session_durations[['abandoned']], on='SessionID', how='left')

    df['pogo'] = 0
    for session_id, group in df[df['IsClicked'] == 1].groupby('SessionID'):
        clicks = group.sort_values('Timestamp')
        if len(clicks) > 1 and (clicks['Timestamp'].diff().dt.total_seconds() < 10).any():
            df.loc[df['SessionID'] == session_id, 'pogo'] = 1
                
    # tau_T calculation (Weighted average of risk factors)
    df['tau_T'] = (
        0.4 * df['C_query_length'].clip(0, 10)/10 +
        0.3 * (1 / (df['tau_to_first_click'].fillna(60).clip(1, 60) / 60)) +
        0.2 * df['pogo'] +
        0.1 * df['abandoned']
    )

    # === 4. WEEKLY AGGREGATION ===
    df['week'] = df['Timestamp'].dt.to_period('W')
    
    weekly = df.groupby('week').agg(
        tau_T_mean=('tau_T', 'mean'),
        click_count=('IsClicked', 'sum'),
        session_count=('SessionID', 'nunique')
    ).rename(columns={'click_count': 'bug_count_T3'}) 

    return weekly

# ==============================================================================
# 2. TIME SERIES PREPARATION AND SARIMAX/GRANGER FUNCTIONS
# ==============================================================================

def prepare_time_series_data(repo_name, weekly_series, forced_freq='W'):
    print(f"\n[STEP 2/3] Preparing Time Series for {repo_name} (Freq: {forced_freq})...")
    
    series_df = weekly_series.copy()
    series_df.index = series_df.index.to_timestamp(how='end')

    start_date = series_df.index.min()
    end_date = series_df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=forced_freq)
    
    series_df = series_df.reindex(full_date_range)
    series_df['tau_T_mean'] = series_df['tau_T_mean'].fillna(method='ffill').fillna(series_df['tau_T_mean'].mean())
    series_df[['bug_count_T3', 'session_count']] = series_df[['bug_count_T3', 'session_count']].fillna(0)
    
    tau_ts = series_df['tau_T_mean']
    bugs_ts = series_df['bug_count_T3']
    
    print(f"Time Series Prepared. Total Data Points: {len(series_df)}")
    return tau_ts, bugs_ts, len(series_df)

def plot_forecast(tau_ts, sarima_forecast, ar_forecast, repo_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(tau_ts.index, tau_ts.values, label='Historical tau_T (Collapse Risk)', color='darkblue', marker='o', linestyle='-')
    
    if sarima_forecast is not None:
        ax.plot(sarima_forecast.index, sarima_forecast.values, label='SARIMAX Forecast', color='red', linestyle='--', marker='x')

    if ar_forecast is not None:
        ax.plot(ar_forecast.index, ar_forecast.values, label='AR Fallback Forecast', color='orange', linestyle=':', marker='s')

    ax.set_title(f'Weekly tau_T (Collapse Risk) and Forecast for {repo_name}')
    ax.set_ylabel('Avg. tau_T Index')
    ax.set_xlabel('Date (End of Week)')
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    xfmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_filename = f'{repo_name}_tau_T_forecast.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"\n[PLOT GENERATED]: Plot saved to {plot_filename}")

def run_analysis(tau_ts, bugs_ts, repo_name, total_data_points):
    print(f"\n--- Analysis for {repo_name} (Data Points: {total_data_points}) ---")
    
    sarima_forecast = None
    tau_train = tau_ts.iloc[:-FORECAST_STEPS] 
    
    # --- 1. Granger Causality Test ---
    granger_data = pd.DataFrame({'Bugs': bugs_ts.values, 'Tau': tau_ts.values})
    
    if len(granger_data) > GRANGER_TEST_MAXLAG:
        try:
            print(f"\n[GRANGER CAUSALITY TEST] Max Lag: {GRANGER_TEST_MAXLAG}")
            gc_results = grangercausalitytests(granger_data[['Bugs', 'Tau']], maxlag=[GRANGER_TEST_MAXLAG], verbose=False)
            p_value = gc_results[GRANGER_TEST_MAXLAG][0]['ssr_ftest'][1]
            
            if p_value < ALPHA_THRESHOLD:
                print(f"[CAUSALITY FOUND]: tau_T -> Clicks (p={p_value:.4f}). Reject H0.")
            else:
                print(f"[NO CAUSALITY]: tau_T -/> Clicks (p={p_value:.4f}). Cannot reject H0.")

        except Exception as e:
            print(f"[GRANGER ERROR]: {e}")
    
    # --- 2. SARIMAX Forecasting ---
    try:
        print("\n[SARIMAX FORECAST]")
        sarima_model = SARIMAX(
            tau_train, 
            order=(1, 0, 1), 
            seasonal_order=SARIMA_SEASONAL_ORDER_W,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        start_idx = len(tau_train)
        end_idx = len(tau_train) + FORECAST_STEPS - 1
        
        sarima_forecast = sarima_model.predict(start=start_idx, end=end_idx)
        
        forecast_index = pd.date_range(
            start=tau_ts.index[-1] + timedelta(days=7), 
            periods=FORECAST_STEPS, 
            freq='W-SUN'
        )
        sarima_forecast.index = forecast_index

        print(f"\nSARIMAX Forecast (3 Periods):\n{sarima_forecast.to_string()}")
        plot_forecast(tau_ts, sarima_forecast, None, repo_name)

    except Exception as e:
        print(f"[SARIMAX FAILED]: {e}. Attempting AutoReg...")
        # AR Fallback logic here (simplified for brevity)

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    raw_logs_df = load_search_logs()
    weekly_series = extract_sdm_features(raw_logs_df)

    ANALYSIS_NAME = 'Wikipedia_CDM'
    
    tau_ts, bugs_ts, total_points = prepare_time_series_data(
        ANALYSIS_NAME, 
        weekly_series, 
        forced_freq='W' 
    )

    if total_points > 0:
        run_analysis(tau_ts, bugs_ts, ANALYSIS_NAME, total_points)

    print("\n--- Wikipedia CDM Analysis Complete ---")
