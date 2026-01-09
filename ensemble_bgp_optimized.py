# -*- coding: utf-8 -*-
"""
BGP Anomaly Ensemble Scoring Pipeline (Isolation Forest + LSTM Autoencoder)
This script loads pre-trained IF and LSTM models, optimizes ensemble weights,
scores new data using a weighted Z-score fusion method, and generates visualizations.
"""

# Imports
import numpy as np
import pandas as pd
import pickle
import json
import joblib
import tensorflow as tf
import os
import matplotlib.pyplot as plt # NEW IMPORT
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json

# class HeuristicDetector:
#     """Implements deterministic rules from Technical Design Section 10."""
    
#     @staticmethod
#     def score_row(row):
#         score = 0.0
#         reasons = []

#         # 1. Churn Rules
#         if row['total_updates'] > 2000: 
#             score = max(score, 1.0); reasons.append("CRITICAL_CHURN")
#         elif row['total_updates'] > 500:
#             score = max(score, 0.8); reasons.append("HIGH_CHURN")

#         # 2. Path Length Rules
#         if row['path_length'] > 25:
#             score = max(score, 0.8); reasons.append("SEVERE_PATH_LEN")
#         elif row['path_length'] > 15:
#             score = max(score, 0.6); reasons.append("MILD_PATH_LEN")

#         # 3. Withdrawal Ratio Rules
#         if row['withdrawal_ratio'] > 0.9:
#             score = max(score, 0.9); reasons.append("MASS_WITHDRAWAL")
#         elif row['withdrawal_ratio'] > 0.7:
#             score = max(score, 0.7); reasons.append("HIGH_WITHDRAWAL_RATIO")

#         # 4. Flapping Rules
#         if row['flap_count'] > 100:
#             score = max(score, 1.0); reasons.append("CRITICAL_FLAP")
        
#         return score, reasons

class HeuristicDetector:
    """Implements deterministic rules from Technical Design Section 10."""
    @staticmethod
    def score_row(row):
        score, reasons = 0.0, []
        # Churn Rules
        if row['total_updates'] > 2000: score = max(score, 1.0); reasons.append("CRITICAL_CHURN")
        elif row['total_updates'] > 500: score = max(score, 0.8); reasons.append("HIGH_CHURN")
        # Path Rules
        if row['path_length'] > 25: score = max(score, 0.8); reasons.append("SEVERE_PATH_LEN")
        # Withdrawal Rules
        if row['withdrawal_ratio'] > 0.9: score = max(score, 0.9); reasons.append("MASS_WITHDRAWAL")
        return score, reasons

def aggregate_mrt_data(df_raw, window='1min'):
    """Advanced aggregator that 'hunts' for columns and fixes numeric errors."""
    df = df_raw.copy()
    
    # 1. Clean headers and find the timestamp column
    df.columns = df.columns.str.strip()
    found_time_col = None
    
    # Check for a column that contains 10-digit Unix timestamps
    for col in df.columns:
        # We check the first valid entry in the column
        first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
        s_val = str(first_val)
        if s_val.isdigit() and len(s_val) >= 10 and s_val.startswith('1'):
            found_time_col = col
            break
            
    if not found_time_col:
        return pd.DataFrame() # Skip chunk if no time data found

    # 2. FIXED: Robust Numeric Conversion
    # We ensure we are passing the Series, not the whole DF
    df['timestamp'] = pd.to_numeric(df[found_time_col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # 3. Find EntryType (Hunt for 'B' or 'W')
    entry_col = next((c for c in df.columns if df[c].astype(str).str.contains('B|W').any()), None)
    if entry_col:
        df['is_announcement'] = df[entry_col].astype(str).str.contains('B').astype(int)
        df['is_withdrawal'] = df[entry_col].astype(str).str.contains('W').astype(int)
    else:
        df['is_announcement'] = 1
        df['is_withdrawal'] = 0

    # 4. Find ASPath
    path_col = next((c for c in df.columns if 'ASPath' in c or 'path' in c.lower()), None)
    if path_col:
        df['path_len_val'] = df[path_col].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    else:
        df['path_len_val'] = 0

    if df.empty: return pd.DataFrame()

    # 5. Aggregation
    resampler = df.set_index('timestamp').resample(window)
    df_agg = resampler.agg({
        'is_announcement': 'sum',
        'is_withdrawal': 'sum',
        'path_len_val': 'mean'
    }).rename(columns={'is_announcement': 'announcements', 'is_withdrawal': 'withdrawals', 'path_len_val': 'path_length'})
    
    df_agg['total_updates'] = df_agg['announcements'] + df_agg['withdrawals']
    df_agg['withdrawal_ratio'] = df_agg['withdrawals'] / df_agg['announcements'].replace(0, 1)
    df_agg['message_rate'] = df_agg['total_updates'] / 60
    
    return df_agg.fillna(0).reset_index()
# --- GLOBAL CONFIGURATION (From bgp_lstm_pipeline.py) ---

# def calculate_nine_features(df, window='1T'):  # CHANGED DEFAULT TO '1T'
#     """Calculates the 9 required features for the BGP anomaly pipeline."""
    
#     # Ensure the timestamp is the index for time-series operations
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df = df.set_index('timestamp').sort_index()
    
#     # Define window size in minutes for calculation (1 minute)
#     # This must match the 'window' parameter (e.g., if window='1T', window_in_minutes = 1)
#     # If using '1T', window_in_minutes is 1.0
#     # If using '5T', window_in_minutes would be 5.0
#     window_in_minutes = 1.0 # ASSUMING '1T' means a 1-minute window
    
#     # 1. Announcements (Using updates_received as proxy)
#     df['announcements'] = df['updates_received']
    
#     # 2. Withdrawals (Using withdrawals_received as proxy)
#     df['withdrawals'] = df['withdrawals_received']
    
#     # 3. Total Updates
#     df['total_updates'] = df['announcements'] + df['withdrawals']
    
#     # Group by peer for rolling calculations
#     grouped = df.groupby('peer_addr')

#     # Calculate rolling sums for message counting
#     rolling_sums = grouped[['announcements', 'withdrawals', 'total_updates']].rolling(window=window).sum() # Added total_updates
#     rolling_sums = rolling_sums.reset_index(level=0, drop=True)
    
#     # Merge rolling sums back to the main DataFrame
#     df = df.join(rolling_sums, rsuffix='_sum') 


#     # 4. Withdrawal Ratio (withdrawals / max(announcements, 1))
#     # Using the calculated rolling sum columns
#     df['withdrawal_ratio'] = df['withdrawals_sum'] / df['announcements_sum'].apply(lambda x: max(x, 1))

#     # 5. Flap Count (Rolling count of significant route changes)
#     df['flap_count'] = grouped['total_updates'].rolling(window=window).apply(
#         lambda x: (x > x.mean() + x.std()).sum(), raw=False).fillna(0)
    
#     # 6. Path Length (Proxy: Rolling Mean of routes_active)
#     df['path_length'] = grouped['routes_active'].rolling(window=window).mean().reset_index(level=0, drop=True)
    
#     # 7. Unique Peers (Scalar count)
#     unique_peers_count = df['peer_addr'].nunique()
#     df['unique_peers'] = unique_peers_count
    
#     # 8. Message Rate (Total Messages over the window, divided by the window length in minutes)
#     df['message_rate'] = df['total_updates_sum'] / window_in_minutes 

#     # 9. Session Resets (Proxy: High total update variance)
#     df['session_resets'] = grouped['total_updates'].rolling(window=window).std().reset_index(level=0, drop=True)
    
#     # Drop rows with NaN (from rolling calculations at the start of the window)
#     df = df.dropna()
    
#     # Select only the features needed for the model (the 9 feature columns)
#     feature_cols = ['announcements', 'withdrawals', 'total_updates', 'withdrawal_ratio', 
#                     'flap_count', 'path_length', 'unique_peers', 'message_rate', 'session_resets']

#     return df[feature_cols].reset_index()
# def calculate_nine_features(df, window='1T'):
#     """
#     SIMPLIFIED: Adjusts DataFrame structure for time-series analysis.
#     This version assumes the bmp_generator has ALREADY inserted the 9 final features.
#     It removes the redundant and conflicting rolling-window feature calculation.
#     """
    
#     # 1. Handle Timestamp (REQUIRED)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
    
#     # 2. Set Index (REQUIRED for time-series alignment)
#     df = df.set_index('timestamp').sort_index()
    
#     # NOTE: The feature calculation logic is removed here because 
#     # the generator inserted 'announcements', 'withdrawals', etc., directly.
    
#     return df
# 

def calculate_nine_features(df):
    """
    Robust Feature Preparation: Maps CSV names and applies defensive filling.
    Ensures no 'zero-stuffing' for state metrics.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- Step 1: Map CSV aliases to Model Feature names ---
    mapping = {
        'estimated_flaps': 'flap_count',
        'avg_path_length': 'path_length',
        'unique_prefixes': 'unique_peers', # Mapping peer-proxy
        'updates_received': 'announcements',
        'withdrawals_received': 'withdrawals'
    }
    for old_col, new_col in mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # --- Step 2: Initialize required columns if totally missing ---
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan # Start with NaN to allow proper filling

    # --- Step 3: Calculate derived metrics ---
    df['total_updates'] = df['announcements'].fillna(0) + df['withdrawals'].fillna(0)
    df['withdrawal_ratio'] = (df['withdrawals'].fillna(0) / 
                              df['announcements'].replace(0, 1).fillna(1)).clip(0, 1)

    # --- Step 4: Defensive Filling (No 0-stuffing) ---
    # State-based: Assume network state persists [ffill]
    state_metrics = ['path_length', 'unique_peers']
    for col in state_metrics:
        df[col] = df[col].ffill().fillna(0) 

    # Rate-based: Fill gaps with median to avoid anomaly spikes [median]
    rate_metrics = ['flap_count', 'message_rate', 'session_resets']
    for col in rate_metrics:
        median_val = df[col].median() if not df[col].dropna().empty else 0
        df[col] = df[col].fillna(median_val)

    # --- Step 5: Final numeric conversion ---
    df = df.set_index('timestamp').sort_index()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

FEATURES = [
    "announcements",
    "withdrawals",
    "total_updates",
    "withdrawal_ratio",
    "flap_count",
    "path_length",
    "unique_peers",
    "message_rate",
    "session_resets",
]
DEFAULT_SEQ_LEN = 10
RANDOM_STATE = 42

# --- ARTIFACT PATHS (CORRECTED & ROBUST) ---
# Define the two separate root directories based on your file structure
IF_ARTIFACTS_ROOT = 'model_artifacts' 
LSTM_ARTIFACTS_ROOT = 'model_output' 

# File variables must contain paths RELATIVE to their respective root, or just the filename.

# LSTM Artifacts (Path is relative to LSTM_ARTIFACTS_ROOT: 'model_output')
LSTM_WEIGHTS_FILE = 'lstm/lstm_final.h5' 
LSTM_CONFIG_FILE = 'pipeline_artifacts_full.pkl' 
SCALER_FILE = 'scaler.pkl' 

# IF Artifacts (Path is relative to IF_ARTIFACTS_ROOT: 'model_artifacts')
IF_MODEL_FILE = 'iso_forest_bgp_production.pkl' # Filename only
FEATURE_FILE = 'bgp_features.pkl'             # Filename only

# --- VISUALIZATION OUTPUT DIRECTORY ---
PLOT_OUTPUT_DIR = 'ensemble_plots'

# -----------------------------------------------------------
# UTILITY FUNCTIONS (Copied from bgp_lstm_pipeline.py for robustness)
# -----------------------------------------------------------
# ... (load_csv_to_df, extract_bgp_features, create_sequences, sequence_reconstruction_error remain here) ...
# [Note: I am omitting the bodies of the utility functions for brevity, 
# but they are present in your full script as provided previously.]
def load_csv_to_df(csv_path, parse_dates=["timestamp"]):
    """Load CSV into pandas DataFrame. Expects a 'timestamp' column."""
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.to_datetime(pd.Series(range(len(df))), unit='s') 
    return df

def extract_bgp_features(df):
    """
    Ensure the 9 BGP features are present and computed if needed.
    (Simplified feature extraction from bgp_lstm_pipeline.py)
    """
    df = df.copy()
    df["total_updates"] = df.get("total_updates", df.get("announcements", 0).fillna(0) + df.get("withdrawals", 0).fillna(0))
    df["withdrawal_ratio"] = df.get("withdrawal_ratio", df.get("withdrawals", 0).fillna(0) / (df.get("announcements", 0).replace(0, 0).fillna(0) + 1e-9)).clip(0, 1)
    df["flap_count"] = df.get("flap_count", df.get("estimated_flaps", 0)).fillna(0)
    df["path_length"] = df.get("path_length", df.get("avg_path_length", 0)).fillna(0)
    df["unique_peers"] = df.get("unique_peers", df.get("unique_nexthops", 0)).fillna(0)
    df["message_rate"] = df.get("message_rate", df["total_updates"].fillna(0) * 60)
    df["session_resets"] = df.get("session_resets", 0).fillna(0)
    
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col].fillna(0), errors="coerce").fillna(0)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    keep_cols = ["timestamp"] + FEATURES
    if "peer_addr" in df.columns:
        keep_cols.append("peer_addr")
        
    return df[keep_cols]

def create_sequences(X, seq_len=DEFAULT_SEQ_LEN, stride=1):
    """Create sliding window sequences from X (numpy array, shape (n, features))."""
    sequences = []
    idx_map = []
    n = len(X)
    for start in range(0, n - seq_len + 1, stride):
        sequences.append(X[start : start + seq_len])
        idx_map.append(start + seq_len - 1)
    return np.array(sequences), np.array(idx_map)

def sequence_reconstruction_error(model, sequences):
    """Return per-sequence Mean Absolute Error (MAE) across timesteps+features"""
    recon = model.predict(sequences, verbose=0)
    errors = np.mean(np.abs(sequences - recon), axis=(1, 2))
    return errors
# -----------------------------------------------------------
# CORE ENSEMBLE FUNCTIONS (load_all_artifacts, optimize_weights, apply_ensemble_scoring remain here)
# -----------------------------------------------------------
def load_all_artifacts(artifacts_dir_unused):
    """Loads all models and configs using the defined dual root paths."""
    
    # 1. Load IF Model (Joins the correct IF root with the IF filename)
    if_path = os.path.join(IF_ARTIFACTS_ROOT, IF_MODEL_FILE)
    iso_forest = joblib.load(if_path) 
    
    feature_path = os.path.join(IF_ARTIFACTS_ROOT, FEATURE_FILE)
    if_features = joblib.load(feature_path)
    
    # 2. Load LSTM Artifacts (Joins the correct LSTM root with its paths)
    pkl_path = os.path.join(LSTM_ARTIFACTS_ROOT, LSTM_CONFIG_FILE)
    
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
        
    # If the scaler is saved separately from the single PKL:
    if "scaler" not in artifacts:
        scaler_path = os.path.join(LSTM_ARTIFACTS_ROOT, SCALER_FILE)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = artifacts["scaler"]

    seq_len = artifacts["seq_len"]
    
    # Reconstruct the model
    model = tf.keras.models.model_from_json(artifacts["model_architecture_json"])
    
    # Load weights
    weights_path = os.path.join(LSTM_ARTIFACTS_ROOT, LSTM_WEIGHTS_FILE) 
    
    if not os.path.exists(weights_path):
        # Fail gracefully if weights are missing
        raise FileNotFoundError(f"LSTM Weights not found at: {weights_path}. Check path integrity.")
        
    model.load_weights(weights_path)
    
    return iso_forest, model, scaler, seq_len

def optimize_weights(df_val_scaled, df_val_feat_if, lstm_model, iso_forest, seq_len):
    """
    Optimizes ensemble weights (w_if, w_lstm) by minimizing the standard deviation
    of the top 10% of ensemble scores on the validation set.
    """
    
    # 1. Calculate Scores on Validation Set
    val_seqs, _ = create_sequences(df_val_scaled, seq_len=seq_len)
    lstm_errors_val_raw = sequence_reconstruction_error(lstm_model, val_seqs)
    if_scores_val_raw = -iso_forest.decision_function(df_val_feat_if.values)
    
    # Align and trim scores
    lstm_errors_val = lstm_errors_val_raw
    if_scores_val = if_scores_val_raw[seq_len - 1:]

    if len(lstm_errors_val) != len(if_scores_val):
        print(f"Warning: Score lengths mismatch after alignment. LSTM={len(lstm_errors_val)}, IF={len(if_scores_val)}. Using minimum length.")
        min_len = min(len(lstm_errors_val), len(if_scores_val))
        lstm_errors_val = lstm_errors_val[:min_len]
        if_scores_val = if_scores_val[:min_len]
    
    # 2. Z-Score Normalization
    z_lstm = (lstm_errors_val - np.mean(lstm_errors_val)) / np.std(lstm_errors_val)
    z_if = (if_scores_val - np.mean(if_scores_val)) / np.std(if_scores_val)

    # 3. Optimization Loop
    best_std = np.inf
    w_if_opt, w_lstm_opt = 0.5, 0.5 # Default weights

    # Search space for weights (0.1 to 1.0, 10 steps)
    for w_if in np.linspace(0.1, 1.0, 10):
        for w_lstm in np.linspace(0.1, 1.0, 10):
            ensemble_score = (w_if * z_if + w_lstm * z_lstm) / (w_if + w_lstm)
            
            # Target metric: Minimize std dev of the top 10% of scores
            top_10_percentile = np.percentile(ensemble_score, 90)
            top_scores = ensemble_score[ensemble_score >= top_10_percentile]
            
            if len(top_scores) > 0:
                current_std = np.std(top_scores)
                if current_std < best_std:
                    best_std = current_std
                    w_if_opt, w_lstm_opt = w_if, w_lstm
    
    print(f"Weight optimization complete. Best StdDev: {best_std:.4f}")
    print(f"Optimized Weights: IF ({w_if_opt:.2f}), LSTM ({w_lstm_opt:.2f})")
    
    return w_if_opt, w_lstm_opt, lstm_errors_val, if_scores_val

def apply_ensemble_scoring(df_all, lstm_model, iso_forest, scaler, w_if_opt, w_lstm_opt, seq_len):
    """
    Applies the full scoring pipeline. 
    Returns: results_df, thresholds, AND idx_map for Heuristic alignment.
    """
    
    # 1. Prepare Data using the updated defensive feature engine
    df_feat = calculate_nine_features(df_all)
    X_all = df_feat[FEATURES].values
    X_all_scaled = scaler.transform(X_all)
    
    # 2. Calculate Raw Scores
    all_seqs, idx_map = create_sequences(X_all_scaled, seq_len=seq_len)
    lstm_errors_raw = sequence_reconstruction_error(lstm_model, all_seqs)
    
    # Isolation Forest needs raw feature values (as per typical training)
    if_scores_raw = -iso_forest.decision_function(X_all)
    
    # 3. Align and Trim Scores
    lstm_errors = lstm_errors_raw
    if_scores_all = if_scores_raw[seq_len - 1:]

    N_sequences = len(lstm_errors)
    if_scores = if_scores_all[:N_sequences]

    # 4. Z-Score Normalization (Standardizes different model scales)
    z_lstm = (lstm_errors - np.mean(lstm_errors)) / (np.std(lstm_errors) + 1e-9)
    z_if = (if_scores - np.mean(if_scores)) / (np.std(if_scores) + 1e-9)
    
    # 5. Calculate Weighted Ensemble Score
    ensemble_score = (w_if_opt * z_if + w_lstm_opt * z_lstm) / (w_if_opt + w_lstm_opt)

    # 6. Apply Severity Logic based on Technical Design percentiles
    threshold_base = np.percentile(ensemble_score, 95)
    thresholds = {
        'low': threshold_base,
        'medium': threshold_base * 1.15,
        'high': threshold_base * 1.35,
        'critical': threshold_base * 1.60,
    }
    
    severity = pd.Series(ensemble_score).apply(lambda s: 
        'CRITICAL' if s > thresholds['critical'] else 
        'HIGH' if s > thresholds['high'] else 
        'MEDIUM' if s > thresholds['medium'] else 
        'LOW' if s > thresholds['low'] else 'NORMAL'
    ).values
    
    # 7. Confidence Logic (Agreement Boost)
    lstm_flag = lstm_errors > np.percentile(lstm_errors, 95) 
    if_flag = if_scores > np.percentile(if_scores, 95) 
    both_agree = (lstm_flag[:N_sequences] & if_flag).astype(int)
    
    score_magnitude = np.clip((ensemble_score - threshold_base) / (threshold_base + 1e-9), 0, 1) * 70 
    agreement_boost = both_agree * 30 
    confidence_scores = np.clip(score_magnitude + agreement_boost, 0, 100)

    # 8. Final Results Assembly
    timestamps = df_feat.index.values[idx_map]
    
    # Defensive check for peer_addr
    if "peer_addr" in df_feat.columns:
        peer_addresses = df_feat["peer_addr"].iloc[idx_map].values
    else:
        peer_addresses = ["Unknown"] * len(idx_map)
    
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'peer_addr': peer_addresses,
        'ensemble_score': ensemble_score,
        'lstm_error': lstm_errors,
        'if_score': if_scores,
        'severity': severity,
        'confidence': confidence_scores,
        'both_agree': both_agree,
    }).sort_values(by=['timestamp']).reset_index(drop=True)

    # RETURN idx_map so run_anomaly_scoring can use it for Heuristics
    return results_df, thresholds, idx_map

# -----------------------------------------------------------
# VISUALIZATION FUNCTION (NEW)
# -----------------------------------------------------------

def generate_visualizations(results_df, thresholds, output_dir=PLOT_OUTPUT_DIR):
    """
    Generates and saves three key visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color map for severity levels
    severity_colors = {
        'CRITICAL': '#e31a1c', # Red
        'HIGH': '#ff7f00',     # Orange
        'MEDIUM': '#fdbf6f',   # Light Orange
        'LOW': '#a6cee3',      # Light Blue
        'NORMAL': '#33a02c'    # Green
    }
    
    # --- Plot 1: Ensemble Score Distribution with Severity Zones ---
    plt.figure(figsize=(10, 6))
    
    # Histogram of scores
    results_df['ensemble_score'].plot(kind='hist', bins=50, density=True, color='gray', alpha=0.6)
    
    # Plot severity thresholds as vertical lines
    threshold_names = ['low', 'medium', 'high', 'critical']
    threshold_values = [thresholds[name] for name in threshold_names]
    
    for val, name in zip(threshold_values, threshold_names):
        plt.axvline(val, color=severity_colors[name.upper()], linestyle='--', linewidth=2, 
                    label=f'{name.capitalize()} ({val:.2f})')
        
    plt.title('Distribution of Ensemble Anomaly Scores with Severity Thresholds')
    plt.xlabel('Weighted Z-Score (Ensemble Score)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_score_distribution.png'))
    plt.close()
    print(f"Saved: 1_score_distribution.png to {output_dir}")

    # --- Plot 2: Confidence Score Distribution (for alerts only) ---
    alerts_df = results_df[results_df['severity'] != 'NORMAL']
    if not alerts_df.empty:
        plt.figure(figsize=(10, 6))
        
        # Plot only scores > 0
        alerts_df['confidence'].plot(kind='hist', bins=20, color='#1f78b4', edgecolor='black')
        
        # Highlight high-confidence region
        plt.axvline(70, color='r', linestyle=':', label='High Confidence (70+)')
        
        plt.title('Confidence Score Distribution for Detected Alerts')
        plt.xlabel('Confidence Score (0-100)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_confidence_distribution.png'))
        plt.close()
        print(f"Saved: 2_confidence_distribution.png to {output_dir}")
    else:
        print("Skipping Confidence Distribution: No alerts detected.")


    # --- Plot 3: Time Series of Ensemble Score with Severity Coloring ---
    plt.figure(figsize=(14, 7))
    
    # Sort for consistent plotting (already done, but good practice)
    results_df = results_df.sort_values('timestamp')
    
    # Plot normal points first (in the background)
    normal_points = results_df[results_df['severity'] == 'NORMAL']
    plt.plot(normal_points['timestamp'], normal_points['ensemble_score'], 
             color=severity_colors['NORMAL'], marker='.', linestyle='', markersize=3, label='NORMAL')

    # Plot anomalies, colored by severity, on top
    alert_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    for level in alert_levels:
        level_points = results_df[results_df['severity'] == level]
        if not level_points.empty:
            plt.scatter(level_points['timestamp'], level_points['ensemble_score'], 
                        color=severity_colors[level], marker='o', s=25, label=level, zorder=3)
    
    plt.title('Ensemble Score Time Series with Severity')
    plt.xlabel('Timestamp')
    plt.ylabel('Ensemble Score (Weighted Z-Score)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_time_series_severity.png'))
    plt.close()
    print(f"Saved: 3_time_series_severity.png to {output_dir}")


# -----------------------------------------------------------
# MAIN EXECUTION BLOCK (Updated)
# -----------------------------------------------------------

# if __name__ == '__main__':
    
#     # 🚨 STEP 1: SET YOUR DATA PATHS 🚨
#     CSV_PATH = 'data/training_data_fast.csv' 
    
#     # The ARTIFACTS_DIR is no longer used for path joining, but kept for reference
#     ARTIFACTS_DIR = 'model_artifacts' 

#     print("--- STARTING ENSEMBLE PIPELINE ---")
    
#     # --- Load Models and Config ---
#     iso_forest, lstm_model, scaler, seq_len = load_all_artifacts(ARTIFACTS_DIR)
#     print(f"✅ Models loaded. Sequence Length: {seq_len}")

#     # --- Load Data ---
#     df_full = load_csv_to_df(CSV_PATH)
#     print(f'Data shape: {df_full.shape}')

#     # --- Split for Weight Optimization ---
#     train_size = int(len(df_full) * 0.7)
#     df_train_raw = df_full.iloc[:train_size].reset_index(drop=True)
#     df_val_raw = df_full.iloc[train_size:].reset_index(drop=True)
#     print(f'Train for optimization: {len(df_train_raw)}, Validation for optimization: {len(df_val_raw)}')
    
#     # --- Prepare Validation Data for Optimization ---
#     df_val_feat = extract_bgp_features(df_val_raw)
#     X_val = df_val_feat[FEATURES].values
#     X_val_scaled = scaler.transform(X_val)
    
#     # --- Optimize Weights (Z-Score Fusion) ---
#     w_if_opt, w_lstm_opt, _, _ = optimize_weights(
#         X_val_scaled, df_val_feat[seq_len - 1:][FEATURES], lstm_model, iso_forest, seq_len
#     )
#     w_if_opt = 0.5 
#     w_lstm_opt = 0.5
#     print("\n--- FORCING EQUAL WEIGHTS (0.5/0.5) FOR ENSEMBLE DIAGNOSTICS ---")
#     print(f"Optimized Weights: IF ({w_if_opt:.2f}), LSTM ({w_lstm_opt:.2f})")

#     # --- Apply Final Scoring to Full Data ---
#     print("\n--- Applying Ensemble Scoring to Full Data ---")
#     results_df, threshold_critical, threshold_high, threshold_medium, threshold_low, both_flagged = \
#         apply_ensemble_scoring(df_full, lstm_model, iso_forest, scaler, w_if_opt, w_lstm_opt, seq_len)
        
#     thresholds = {
#         'critical': threshold_critical, 'high': threshold_high, 
#         'medium': threshold_medium, 'low': threshold_low
#     }

#     # --- Save Results, Config, and Generate Plots (NEW) ---
    
#     # 1. Save alerts CSV
#     alerts_df = results_df[results_df['severity'] != 'NORMAL']
#     alerts_df.to_csv('ensemble_alerts.csv', index=False)
#     print(f'\nSaved: ensemble_alerts.csv ({len(alerts_df)} alerts)')

#     # 2. Save optimized config
#     config = {
#         'version': '3.0_optimized_visualized',
#         'weights': {'if': float(w_if_opt), 'lstm': float(w_lstm_opt)},
#         'thresholds': {k: float(v) for k, v in thresholds.items()},
#         'metrics': {
#             'total_samples': int(len(results_df)),
#             'critical_count': int((results_df['severity'] == 'CRITICAL').sum()),
#             'high_count': int((results_df['severity'] == 'HIGH').sum()),
#             'model_agreement_count': int(both_flagged.sum()),
#             'high_confidence_count': int((results_df['confidence'] > 70).sum())
#         }
#     }
#     with open('ensemble_config_optimized.json', 'w') as f:
#         json.dump(config, f, indent=2)
#     print('Saved: ensemble_config_optimized.json')
#     # 3. Generate Visualizations
#     print("\n--- Generating Visualizations ---")
#     generate_visualizations(results_df, thresholds)

#     # --- Final Report ---
#     print('\n=== ENSEMBLE SUMMARY REPORT ===')
#     # ... (rest of the report logic)

# ... (all existing imports, configs, and functions, including apply_ensemble_scoring) ...

# def run_anomaly_scoring(df_raw: pd.DataFrame) -> pd.DataFrame:
#     """
#     Main integrated scoring function. 
#     Implements Defensive Feature Engineering, ML Ensemble, and Heuristic Layer.
#     """
#     # 1. Load All Artifacts
#     try:
#         # artifacts_dir_unused=None as per your existing logic
#         iso_forest, lstm_model, scaler, seq_len = load_all_artifacts(None)
#         print("✅ Models and Scaler loaded successfully.")
#     except Exception as e:
#         print(f"🛑 CRITICAL ERROR: Failed to load ML artifacts. Error: {e}")
#         return pd.DataFrame()

#     # 2. Defensive Feature Engineering (FIXES THE KEYERROR)
#     # This function handles the mapping of 'estimated_flaps' -> 'flap_count', etc.
#     print("--- Preparing Defensive Features (Addressing Missing Data) ---")
#     df_feat = calculate_nine_features(df_raw)
    
#     # 3. Prepare Validation Data for Logic Flow (Internal Pipeline requirement)
#     # We use df_feat here because it contains the guaranteed FEATURES list
#     val_size = max(seq_len * 2, int(len(df_feat) * 0.3))
#     df_val_feat = df_feat.iloc[:val_size]
    
#     X_val = df_val_feat[FEATURES].values
#     X_val_scaled = scaler.transform(X_val)

#     # 4. Handle Weights
#     # We maintain the optimization call to satisfy the pipeline structure
#     # but hardcode to 50/50 as per your previous requirement.
#     w_if_opt, w_lstm_opt = 0.5, 0.5
#     print(f"--- USING FIXED WEIGHTS: IF ({w_if_opt:.2f}), LSTM ({w_lstm_opt:.2f}) ---")
    
#     # 5. Apply Ensemble Scoring (ML LAYER)
#     # Ensure your 'apply_ensemble_scoring' is updated to return idx_map!
#     print("\n--- Applying ML Ensemble Scoring ---")
#     results_df, thresholds, idx_map = apply_ensemble_scoring(
#         df_raw,  # We pass raw, but the internal function will call feature engine
#         lstm_model, 
#         iso_forest, 
#         scaler, 
#         w_if_opt, 
#         w_lstm_opt, 
#         seq_len
#     )

#     # 6. Apply Heuristic Detection (DETERMINISTIC LAYER)
#     print("--- Applying Heuristic Detection Layer ---")
#     h_scores = []
#     h_reasons = []
#     # Align the feature dataframe with the windows produced by the ML models
#     df_feat_aligned = df_feat.iloc[idx_map] 

#     for _, row in df_feat_aligned.iterrows():
#         score, reasons = HeuristicDetector.score_row(row)
#         h_scores.append(score)
#         h_reasons.append("|".join(reasons) if reasons else "NORMAL")

#     results_df['heuristic_score'] = h_scores
#     results_df['heuristic_reasons'] = h_reasons

#     # 7. Correlation Engine (DEFENSE IN DEPTH)
#     # Escalates severity based on agreement between sources
#     def correlate_severity(row):
#         sources = 0
#         # Check Heuristic trigger
#         if row['heuristic_score'] > 0.6: sources += 1
#         # Check Ensemble ML trigger
#         if row['ensemble_score'] > thresholds['medium']: sources += 1
#         # Check for model agreement (IF + LSTM)
#         if row.get('both_agree', 0) == 1: sources += 1 
        
#         # Escalation Logic from Technical Design
#         if sources >= 3: return "CRITICAL"
#         if sources == 2: return "HIGH"
        
#         # Fallback to the original ensemble severity if only one source triggered
#         return row['severity']

#     results_df['final_severity'] = results_df.apply(correlate_severity, axis=1)
    
#     print(f"✅ Integrated Scoring complete. Multi-source alerts: {len(results_df[results_df['final_severity'] != 'NORMAL'])}")
    
#     return results_df, thresholds


def run_anomaly_scoring(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Main integrated scoring function. 
    Handles Raw (MRT) vs. Aggregated data and forces lowercase 'timestamp'.
    """
    # --- STEP 1: LOAD MODELS ---
    try:
        # artifacts_dir_unused=None as per your existing logic
        iso_forest, lstm_model, scaler, seq_len = load_all_artifacts(None)
        print("✅ Models and Scaler loaded successfully.")
    except Exception as e:
        print(f"🛑 CRITICAL ERROR: Failed to load ML artifacts. Error: {e}")
        return pd.DataFrame(), {}

    # --- STEP 2: DATA PREPARATION & NORMALIZATION ---
    df_work = df_input.copy()

    # CRITICAL: Normalize 'Timestamp' to 'timestamp' immediately to prevent KeyErrors
    if 'Timestamp' in df_work.columns:
        df_work = df_work.rename(columns={'Timestamp': 'timestamp'})
    elif 'timestamp' not in df_work.columns:
        # If it's your raw CSV, the column might be 'Timestamp' or index 1
        df_work = df_work.rename(columns={df_work.columns[1]: 'timestamp'})
    
    # Detect if this is Raw event data (contains PeerIP or ASPath)
    is_raw = 'PeerIP' in df_work.columns or 'ASPath' in df_work.columns
    
    if is_raw:
        print("📊 Raw MRT data detected. Converting to 1-minute snapshots...")
        df_prepared = aggregate_mrt_data(df_work, window='1min') 
    else:
        df_prepared = df_work

    # STEP 3: DEFENSIVE FEATURE ENGINEERING
    df_feat = calculate_nine_features(df_prepared)
    
    # STEP 4: ML ENSEMBLE SCORING
    print("--- Applying ML Ensemble Scoring ---")
    results_df, thresholds, idx_map = apply_ensemble_scoring(
        df_prepared, lstm_model, iso_forest, scaler, 0.5, 0.5, seq_len
    )

    # STEP 5: HEURISTIC LAYER
    h_scores, h_reasons = [], []
    df_feat_aligned = df_feat.iloc[idx_map] 

    for _, row in df_feat_aligned.iterrows():
        score, reasons = HeuristicDetector.score_row(row)
        h_scores.append(score)
        h_reasons.append("|".join(reasons) if reasons else "NORMAL")

    results_df['heuristic_score'] = h_scores
    results_df['heuristic_reasons'] = h_reasons

    # STEP 6: FINAL CORRELATION ENGINE
    def correlate_severity(row):
        sources = 0
        if row['heuristic_score'] > 0.6: sources += 1
        if row['ensemble_score'] > thresholds['medium']: sources += 1
        if row.get('both_agree', 0) == 1: sources += 1 
        
        if sources >= 3: return "CRITICAL"
        if sources == 2: return "HIGH"
        return row['severity']

    results_df['final_severity'] = results_df.apply(correlate_severity, axis=1)
    
    print(f"✅ Integrated Scoring complete. Generated {len(results_df)} sequences.")
    return results_df, thresholds
    
    # ... (Final print statement and return remain unchanged) ...
    # print(f"✅ Scoring complete. Total alerts generated: {len(results_df[results_df['severity'] != 'NORMAL'])}")
    
    # # 3. Visualization/Config (Optional - can be run separately)
    # # The visualization logic (generate_visualizations) should be moved outside
    # # of this core function if you want a clean, scheduled pipeline.
    
    # # Return the results
    # return results_df ,thresholds

# Remove the original if __name__ == '__main__': block
# if __name__ == '__main__':
#    ...