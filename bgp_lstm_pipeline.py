"""
bgp_lstm_pipeline.py
End-to-end unsupervised LSTM Autoencoder pipeline for BGP anomaly detection.

Usage:
  - Import as module and call train_pipeline_from_csv(csv_path, output_dir)
  - Or run as script (add CLI if you want)

The pipeline implements:
1) Data loading and exact feature extraction (9 features)
2) Scaling with StandardScaler
3) Sliding window sequences (sequence_length=10 default)
4) Train/test split (temporal or random), training on filtered "normal" distribution
5) LSTM autoencoder (Keras/TensorFlow)
6) Threshold = 95th percentile of training reconstruction error
7) Severity mapping according to technical doc rules
8) Plots: loss curves, error histogram, error vs time, threshold analysis
9) Save/Load artifacts (model, scaler, threshold, metadata)
10) Utilities for retrain and drift monitoring
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ML libs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import model_from_json

# -------------------------
# CONFIG / FEATURES
# -------------------------
# Exact BGP features from the technical design doc (9 features)
# announcements, withdrawals, total_updates, withdrawal_ratio, flap_count,
# path_length, unique_peers, message_rate, session_resets
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

DEFAULT_SEQ_LEN = 10  # per document: sequence_length = 10
RANDOM_STATE = 42

# -------------------------
# UTILITIES: I/O & helpers
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# -------------------------
# 1) DATA LOADING & FEATURE EXTRACTION
# -------------------------
def load_csv_to_df(csv_path, parse_dates=["timestamp"]):
    """Load CSV into pandas DataFrame. Expects a 'timestamp' column."""
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("CSV must contain a 'timestamp' column")
    return df

def extract_bgp_features(df):
    """
    Ensure the 9 BGP features are present and computed if needed.
    If some are missing, attempt sensible derivations; otherwise fill 0.
    """
    df = df.copy()
    # Derive total_updates if not present
    if "total_updates" not in df.columns:
        if "announcements" in df.columns and "withdrawals" in df.columns:
            df["total_updates"] = df["announcements"].fillna(0) + df["withdrawals"].fillna(0)
        else:
            df["total_updates"] = 0

    # withdrawal_ratio: withdrawals / max(announcements, 1)
    if "withdrawal_ratio" not in df.columns:
        df["withdrawal_ratio"] = df.get("withdrawals", 0).fillna(0) / (df.get("announcements", 0).replace(0, 0).fillna(0) + 1e-9)
        # clip to [0, 1]
        df["withdrawal_ratio"] = df["withdrawal_ratio"].clip(0, 1)

    # flap_count fallback
    if "flap_count" not in df.columns:
        # If no flap count, use estimated_flaps or 0
        if "estimated_flaps" in df.columns:
            df["flap_count"] = df["estimated_flaps"].fillna(0)
        else:
            df["flap_count"] = 0

    # path_length fallback
    if "path_length" not in df.columns:
        # try avg_path_length or avg_as_path_length
        if "avg_path_length" in df.columns:
            df["path_length"] = df["avg_path_length"].fillna(0)
        else:
            df["path_length"] = 0

    # unique_peers fallback
    if "unique_peers" not in df.columns:
        df["unique_peers"] = df.get("unique_peers", df.get("unique_nexthops", 0)).fillna(0)

    # message_rate fallback: if total_updates is per hour, convert to per-minute
    if "message_rate" not in df.columns:
        # assume total_updates is per sample window; if timestamp granularity unknown, fallback to total_updates * 60
        df["message_rate"] = df.get("message_rate", df["total_updates"].fillna(0) * 60)

    # session_resets fallback
    if "session_resets" not in df.columns:
        df["session_resets"] = df.get("session_resets", 0).fillna(0)

    # ensure all feature columns exist and are numeric
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col].fillna(0), errors="coerce").fillna(0)

    # sort by timestamp (important for time splits and sequences)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Keep only necessary columns + timestamp + peer (if present) for future grouping
    keep_cols = ["timestamp"] if "timestamp" in df.columns else []
    keep_cols += ["peer_addr"] if "peer_addr" in df.columns else []
    keep_cols += FEATURES
    return df[keep_cols]

# -------------------------
# 1b) Scaling & sequences
# -------------------------
def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def scale_features(scaler, X):
    return scaler.transform(X)

def create_sequences(X, seq_len=DEFAULT_SEQ_LEN, stride=1):
    """
    Create sliding window sequences from X (numpy array, shape (n, features)).
    Returns array of shape (n_sequences, seq_len, features).
    Attribution note: per design doc, error of sequence i..i+L-1 is attributed to last row i+L-1.
    """
    sequences = []
    idx_map = []  # maps sequence index -> index of the last row in original X
    n = len(X)
    for start in range(0, n - seq_len + 1, stride):
        seq = X[start : start + seq_len]
        sequences.append(seq)
        idx_map.append(start + seq_len - 1)
    return np.array(sequences), np.array(idx_map)

# -------------------------
# 2) TRAIN/TEST SPLIT (UNSUPERVISED)
# -------------------------
def temporal_train_test_split(df, train_ratio=0.8, by_peer=False):
    """
    Temporal split: keep temporal ordering. Optionally split per peer (prevents leakage across peers).
    Returns train_df, test_df
    """
    if not by_peer or "peer_addr" not in df.columns:
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    # split per peer
    trains = []
    tests = []
    for peer in df["peer_addr"].unique():
        dfp = df[df["peer_addr"] == peer].sort_values("timestamp")
        split_idx = int(len(dfp) * train_ratio)
        trains.append(dfp.iloc[:split_idx])
        tests.append(dfp.iloc[split_idx:])
    train_df = pd.concat(trains).sort_values("timestamp").reset_index(drop=True)
    test_df = pd.concat(tests).sort_values("timestamp").reset_index(drop=True)
    return train_df, test_df

def filter_normal_data(df, percentile=95, metrics=None):
    """
    Remove extreme outliers from the training set so we train on 'normal' distribution.
    Default metrics follow the design doc: total_updates, withdrawal_ratio, flap_count.
    This is a conservative unsupervised approach to reduce training contamination.
    """
    if metrics is None:
        metrics = ["total_updates", "withdrawal_ratio", "flap_count"]
    df_filtered = df.copy()
    for m in metrics:
        if m in df_filtered.columns:
            thr = df_filtered[m].quantile(percentile / 100.0)
            df_filtered = df_filtered[df_filtered[m] <= thr]
    return df_filtered.reset_index(drop=True)

# -------------------------
# 3) LSTM AUTOENCODER ARCHITECTURE
# -------------------------
def build_lstm_autoencoder(seq_len, n_features, latent_dim=32, dropout=0.2):
    """
    Builds a symmetric LSTM autoencoder.
    - encoder: LSTM(64)->Dropout->LSTM(latent_dim)
    - decoder: RepeatVector->LSTM(latent_dim)->Dropout->LSTM(64)->TimeDistributed(Dense(n_features))
    Compiled with 'adam' and MSE loss.
    """
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, activation="tanh", return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(latent_dim, activation="tanh", return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(64, activation="tanh", return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------
# 4) TRAINING: loops, callbacks, saving
# -------------------------
def train_lstm_autoencoder(
    sequences,
    seq_len,
    n_features,
    model_dir,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    patience=8,
    latent_dim=32,
    dropout=0.2,
):
    """
    Trains the LSTM autoencoder on sequences (X -> X).
    Returns model, history, and computed threshold (95th percentile).
    Saves checkpoints and history to model_dir.
    """
    ensure_dir(model_dir)

    model = build_lstm_autoencoder(seq_len, n_features, latent_dim=latent_dim, dropout=dropout)
    model.summary()

    chk_path = os.path.join(model_dir, "lstm_best.h5")
    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)
    cb_chk = callbacks.ModelCheckpoint(chk_path, monitor="val_loss", save_best_only=True, verbose=1)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    history = model.fit(
        sequences,
        sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[cb_early, cb_chk, cb_reduce],
        verbose=2,
    )

    # compute reconstruction error per sequence on training sequences
    reconstructions = model.predict(sequences, verbose=0)
    train_errors = np.mean(np.abs(sequences - reconstructions), axis=(1, 2))
    threshold = float(np.percentile(train_errors, 95))  # per doc
    # Save model weights (best saved by checkpoint) and metadata
    model.save(os.path.join(model_dir, "lstm_final.h5"))
    save_json(os.path.join(model_dir, "training_metadata.json"), {"threshold": threshold, "trained_at": datetime.utcnow().isoformat()})

    return model, history, threshold, train_errors

# -------------------------
# 5) ANOMALY DETECTION & SEVERITY MAPPING
# -------------------------
def sequence_reconstruction_error(model, sequences):
    """Return per-sequence mean absolute error across timesteps+features"""
    recon = model.predict(sequences, verbose=0)
    errors = np.mean(np.abs(sequences - recon), axis=(1, 2))
    return errors

def severity_from_error(error, threshold):
    """
    Map a numeric reconstruction error to severity levels according to doc:
    - Low: error <= threshold
    - Medium: threshold < error <= 1.5*threshold
    - High: 1.5*threshold < error <= 2.0*threshold
    - Critical: error > 2.0*threshold
    Returns string severity.
    """
    if error <= threshold:
        return "low"
    elif error <= 1.5 * threshold:
        return "medium"
    elif error <= 2.0 * threshold:
        return "high"
    else:
        return "critical"

def classify_sequences(errors, threshold):
    """Return array of severity labels and boolean anomaly flags (error > threshold)"""
    labels = [severity_from_error(e, threshold) for e in errors]
    flags = (errors > threshold).astype(int)
    return np.array(labels), flags

# -------------------------
# 6) EVALUATION (UNSUPERVISED METRICS & PLOTS)
# -------------------------
def plot_loss(history, out_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("LSTM Autoencoder Loss")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()

def plot_error_histogram(errors, threshold=None, out_path=None):
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=80, alpha=0.8)
    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Histogram")
    if threshold is not None:
        plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()

def plot_error_over_time(df_timestamps, errors, threshold=None, out_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(df_timestamps, errors, marker=".", linestyle="-", alpha=0.7)
    if threshold is not None:
        plt.axhline(threshold, color="red", linestyle="--")
    plt.xlabel("Timestamp")
    plt.ylabel("Sequence reconstruction error")
    plt.title("Reconstruction error vs time (sequence attribution to last timestamp)")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()

def percentage_anomalies(flags):
    return float(np.sum(flags) / len(flags) * 100.0)

def pseudo_roc_auc(errors, top_percent=5):
    """
    Create pseudo-labels by taking top_percent highest errors as anomalies.
    Compute AUC of error scores vs these pseudo-labels.
    This is heuristic — see docs for caveats.
    """
    n = len(errors)
    k = max(1, int(n * top_percent / 100.0))
    # label top-k errors as anomalies
    idx_sorted = np.argsort(errors)[::-1]
    labels = np.zeros(n, dtype=int)
    labels[idx_sorted[:k]] = 1
    try:
        auc = roc_auc_score(labels, errors)
    except Exception:
        auc = None
    return auc, labels

# -------------------------
# 7) SAVE / LOAD ARTIFACTS
# -------------------------
def save_artifacts(output_dir, scaler, model, threshold, feature_names=FEATURES, seq_len=DEFAULT_SEQ_LEN):
    ensure_dir(output_dir)
    # save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    # model: model.save was done in training; ensure saved as 'lstm_final.h5'
    # save config
    config = {
        "feature_names": feature_names,
        "seq_len": seq_len,
        "threshold": float(threshold),
        "saved_at": datetime.utcnow().isoformat(),
    }
    save_json(os.path.join(output_dir, "config.json"), config)
    print(f"Artifacts saved to {output_dir}")
# this is for pkl 
def save_artifacts_to_single_pkl(output_dir, scaler, model, threshold, feature_names=FEATURES, seq_len=DEFAULT_SEQ_LEN):
    """
    Saves all pipeline artifacts (scaler, model config, threshold) into a single PKL file.
    The model weights are saved separately as H5 due to pickle limitations.
    """
    ensure_dir(output_dir)
    
    # 1. Save model weights separately (essential for robust loading)
    #weights_path = os.path.join(output_dir, "lstm_model_weights_for_pkl.h5")
    weights_path = os.path.join(output_dir, "lstm_model_for_pkl.weights.h5")
    model.save_weights(weights_path) 

    # 2. Package all artifacts into a dictionary
    artifacts = {
        "scaler": scaler,
        "threshold": float(threshold),
        "feature_names": feature_names,
        "seq_len": seq_len,
        "model_architecture_json": model.to_json(), # Save architecture as JSON string
        "weights_h5_path": weights_path, # Save the path to the weights
        "saved_at": datetime.utcnow().isoformat(),
    }
    
    # 3. Pickle the dictionary
    pkl_path = os.path.join(output_dir, "pipeline_artifacts_full.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)
        
    print(f"Pipeline artifacts saved to: {pkl_path} and weights to {weights_path}")

def load_artifacts(output_dir):
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    model_path = os.path.join(output_dir, "lstm_final.h5")
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Missing artifacts in output_dir")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model(model_path)
    config = load_json(config_path)
    return scaler, model, config
# this for pkl
def load_artifacts_from_single_pkl(output_dir):
    """
    Loads all pipeline artifacts from a single PKL file and reconstructs the Keras model.
    Returns: scaler, model, config (where config is the dictionary loaded from the pkl)
    """
    pkl_path = os.path.join(output_dir, "pipeline_artifacts_full.pkl")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing single-file artifacts at {pkl_path}")
        
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
        
    # Reconstruct the model from JSON architecture
    model = model_from_json(artifacts["model_architecture_json"])
    
    # Load weights from the separate H5 file
    weights_path = artifacts.get("weights_h5_path")
    if weights_path and os.path.exists(weights_path):
         model.load_weights(weights_path)
    else:
        # Important: If weights are missing, the model will have random initial weights!
        print("Warning: Model weights H5 file not found. Model loaded without trained weights.")

    # The 'config' is now the entire artifacts dictionary
    return artifacts["scaler"], model, artifacts

# -------------------------
# 8) FULL TRAINING PIPELINE ENTRYPOINT
# -------------------------
def train_pipeline_from_csv(
    csv_path,
    output_dir,
    seq_len=DEFAULT_SEQ_LEN,
    train_ratio=0.8,
    by_peer=False,
    filter_percentile=95,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
):
    """
    Full pipeline:
      - load CSV
      - extract features
      - temporal split
      - filter 'normal' data in train partition
      - fit scaler on train
      - create sequences (train & test)
      - train LSTM on train sequences only
      - compute threshold (95th percentile)
      - evaluate on test sequences & produce plots/metrics
      - save artifacts to output_dir
    """
    ensure_dir(output_dir)
    df = load_csv_to_df(csv_path)
    df_feat = extract_bgp_features(df)

    # split
    train_df, test_df = temporal_train_test_split(df_feat, train_ratio=train_ratio, by_peer=by_peer)

    # filter normal in training partition
    train_df_normal = filter_normal_data(train_df, percentile=filter_percentile)

    # features arrays
    X_train = train_df_normal[FEATURES].values
    X_test = test_df[FEATURES].values

    # scaler
    scaler = fit_scaler(X_train)
    X_train_scaled = scale_features(scaler, X_train)
    X_test_scaled = scale_features(scaler, X_test)

    # sequences & index mapping (sequence -> index of last row)
    train_seqs, train_idx_map = create_sequences(X_train_scaled, seq_len=seq_len)
    test_seqs, test_idx_map = create_sequences(X_test_scaled, seq_len=seq_len)

    if len(train_seqs) == 0:
        raise ValueError("Not enough train data to create sequences. Need at least seq_len rows.")

    # Train model
    model_dir = os.path.join(output_dir, "lstm")
    model, history, threshold, train_errors = train_lstm_autoencoder(
        train_seqs,
        seq_len,
        n_features=len(FEATURES),
        model_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    # Save scaler + config + threshold
    #save_artifacts(output_dir, scaler, model, threshold, feature_names=FEATURES, seq_len=seq_len)
    save_artifacts_to_single_pkl(output_dir, scaler, model, threshold, feature_names=FEATURES, seq_len=seq_len)

    # Evaluate on test sequences
    test_errors = sequence_reconstruction_error(model, test_seqs)

    # Map errors to the timestamps of test set: each sequence maps to last row timestamp
    # get timestamps for test_df (must align with indices)
    test_timestamps = test_df["timestamp"].reset_index(drop=True)
    # last-row indices in test partition are given by test_idx_map; map to timestamps
    ts_seq = test_timestamps.iloc[test_idx_map].values if len(test_idx_map) > 0 else []

    # classification & metrics
    labels, flags = classify_sequences(test_errors, threshold)
    pct_anom = percentage_anomalies(flags)
    auc, pseudo_labels = pseudo_roc_auc(test_errors, top_percent=5)  # heuristic
    # Save evaluation artifacts
    eval_obj = {
        "threshold": threshold,
        "pct_anomalies": pct_anom,
        "test_sequences": len(test_seqs),
        "train_sequences": len(train_seqs),
        "pseudo_auc_top5pct": auc,
    }
    save_json(os.path.join(output_dir, "evaluation.json"), eval_obj)

    # Plots
    plot_loss(history, out_path=os.path.join(output_dir, "loss.png"))
    plot_error_histogram(np.concatenate([train_errors, test_errors]), threshold=threshold, out_path=os.path.join(output_dir, "error_hist.png"))
    if len(ts_seq) > 0:
        plot_error_over_time(ts_seq, test_errors, threshold=threshold, out_path=os.path.join(output_dir, "error_over_time.png"))

    # Print summary
    print("=== TRAINING SUMMARY ===")
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")
    print(f"Threshold (95th percentile of train errors): {threshold:.6f}")
    print(f"% anomalies in test set (error > threshold): {pct_anom:.4f}%")
    print(f"Pseudo AUC (top-5% heuristic): {auc}")
    print("Artifacts saved at:", output_dir)

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "eval": eval_obj,
    }

# -------------------------
# 9) PRODUCTION INFERENCE HELPERS
# -------------------------
def score_new_batch(df_new, artifacts_dir):
    """
    Score an incoming dataframe (rows ordered by timestamp). The function:
      - loads scaler & model & config
      - extracts features
      - scales them
      - forms sequences using last (seq_len) rows (sliding)
      - returns per-row severity attached to last row of each sequence
    """
    #scaler, model, config = load_artifacts(artifacts_dir)
    scaler, model, config = load_artifacts_from_single_pkl(artifacts_dir)
    seq_len = config["seq_len"]
    threshold = config["threshold"]

    df_feat = extract_bgp_features(df_new)
    X = df_feat[FEATURES].values
    X_scaled = scaler.transform(X)
    seqs, idx_map = create_sequences(X_scaled, seq_len=seq_len)
    if len(seqs) == 0:
        return pd.DataFrame()  # not enough data to score

    errors = sequence_reconstruction_error(model, seqs)
    severities, flags = classify_sequences(errors, threshold)
    # Attach results to the timestamp corresponding to last row of each sequence
    timestamps = df_feat["timestamp"].reset_index(drop=True).iloc[idx_map].values
    peer_col = df_feat["peer_addr"].reset_index(drop=True).iloc[idx_map].values if "peer_addr" in df_feat.columns else None

    out = pd.DataFrame({
        "timestamp": timestamps,
        "reconstruction_error": errors,
        "severity": severities,
        "anomaly_flag": flags,
    })
    if peer_col is not None:
        out["peer_addr"] = peer_col

    return out

# -------------------------
# 10) RETRAINING & DRIFT MONITORING (GUIDANCE)
# -------------------------
def monitor_feature_drift(train_stats, new_stats, threshold_pct=0.2):
    """
    Simple drift detector: compare historical mean/std vs new batch mean/std
    Returns dict of feature->drift_bool. This is a placeholder; for production use PSI/KL or more advanced tests.
    """
    drift = {}
    for feat in FEATURES:
        if feat in train_stats and feat in new_stats:
            mean_change = abs(train_stats[feat]["mean"] - new_stats[feat]["mean"]) / (abs(train_stats[feat]["mean"]) + 1e-9)
            std_change = abs(train_stats[feat]["std"] - new_stats[feat]["std"]) / (abs(train_stats[feat]["std"]) + 1e-9)
            drift[feat] = (mean_change > threshold_pct) or (std_change > threshold_pct)
    return drift

def compute_basic_stats(df):
    stats = {}
    for feat in FEATURES:
        arr = df[feat].values
        stats[feat] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return stats

def retrain_policy_suggestion():
    """
    Suggestion (documented): retrain weekly or biweekly, or when drift detected:
      - If > 3 features drift beyond 20% vs training stats -> schedule retrain
      - Keep model metadata, dataset snapshot and compare AUC on labelled incidents when available.
    """
    return {
        "retrain_interval": "weekly or biweekly",
        "drift_trigger": ">=3 features drift > 20% mean/std",
        "manual_labeling": "collect labelled incidents into dataset for supervised validation",
    }

# End of module
