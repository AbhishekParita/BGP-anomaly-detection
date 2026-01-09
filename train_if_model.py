import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from bgp_lstm_pipeline import (
    load_csv_to_df,
    extract_bgp_features,
    temporal_train_test_split,
    filter_normal_data,
    ensure_dir,
    FEATURES  # Imports the list of 9 BGP features
)

# --- CONFIGURATION ---
# Typical contamination is 0.001 to 0.01. This is the expected fraction of anomalies.
IF_CONTAMINATION = 0.01 
IF_N_ESTIMATORS = 200
IF_RANDOM_STATE = 42

# --- ISOLATION FOREST PIPELINE ---

def train_and_save_isolation_forest(csv_path, output_dir, contamination=IF_CONTAMINATION):
    """
    Trains an Isolation Forest model using the same pre-processing as the LSTM pipeline.
    
    1. Loads and extracts features.
    2. Splits data temporally.
    3. Filters the training set to retain only 'normal' data (95th percentile).
    4. Trains the Isolation Forest model.
    5. Saves the model and the feature list (required by ensemble_bgp_optimized.py).
    """
    
    ensure_dir(output_dir)
    print(f"Loading and processing data from {csv_path}...")
    
    # 1. Load data and extract features (uses your pipeline functions)
    df = load_csv_to_df(csv_path)
    df_feat = extract_bgp_features(df)

    # 2. Temporal Split (using 80% for training data)
    train_df, _ = temporal_train_test_split(df_feat, train_ratio=0.8, by_peer=False)
    
    # 3. Filter 'normal' data in the training partition (matches LSTM approach)
    train_df_normal = filter_normal_data(train_df, percentile=95)
    
    # Features for IF model (IF does not use sequences or a scaler)
    X_train_if = train_df_normal[FEATURES].values
    
    print(f"Training Isolation Forest on {len(X_train_if)} 'normal' samples...")
    
    # 4. Train Model
    iso_forest = IsolationForest(
        contamination=contamination, 
        random_state=IF_RANDOM_STATE, 
        n_estimators=IF_N_ESTIMATORS,
        n_jobs=-1 # Use all available cores
    )
    iso_forest.fit(X_train_if)
    print("Isolation Forest training complete.")
    
    # 5. Save Artifacts (Matching Ensemble Script Requirements)
    # The ensemble script expects these specific filenames:
    if_model_path = os.path.join(output_dir, 'iso_forest_bgp_production.pkl')
    features_path = os.path.join(output_dir, 'bgp_features.pkl')

    joblib.dump(iso_forest, if_model_path)
    joblib.dump(FEATURES, features_path)
    
    print(f"\n✅ Isolation Forest model saved to: {if_model_path}")
    print(f"✅ Features list saved to: {features_path}")
    
    return iso_forest


if __name__ == '__main__':
    # 🚨 STEP 1: SET YOUR TRAINING DATA PATH 🚨
    # Use the same CSV you used to train your LSTM model.
    TRAINING_CSV_PATH = "data\synthetic_30d.csv" 
    
    # This directory will hold the IF and LSTM artifacts (if you run both training pipelines here)
    OUTPUT_ARTIFACTS_DIR = "model_artifacts" 
    
    print("--- STARTING ISOLATION FOREST TRAINING ---")
    
    trained_if_model = train_and_save_isolation_forest(
        csv_path=TRAINING_CSV_PATH, 
        output_dir=OUTPUT_ARTIFACTS_DIR,
        contamination=0.01 # 1% assumed anomaly rate
    )
    
    print("\n--- NEXT STEP ---")
    print(f"Ensure your LSTM artifacts (lstm_final.h5, scaler.pkl, config.json/pipeline_artifacts_full.pkl) are also in the '{OUTPUT_ARTIFACTS_DIR}' directory.")
    print("Then, run your ensemble_bgp_optimized.py script.")