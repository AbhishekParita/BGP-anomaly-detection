import pandas as pd
import os
import json
from ensemble_bgp_optimized import run_anomaly_scoring, generate_visualizations

# --- CONFIGURATION ---
NEW_DATA_PATH = 'processed_distributed_test_nimda.csv' 
TEST_OUTPUT_DIR = 'testing_run/results'
TEST_PLOT_DIR = 'testing_run/plots/REd'

def evaluate_processed_dataset():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_PLOT_DIR, exist_ok=True)

    print(f"--- Analyzing Processed Dataset: {NEW_DATA_PATH} ---")

    try:
        # Load the pre-aggregated data
        df = pd.read_csv(NEW_DATA_PATH)
        
        # Ensure 'timestamp' column is correctly named and formatted
        if 'Timestamp' in df.columns:
            df = df.rename(columns={'Timestamp': 'timestamp'})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"✅ Loaded {len(df)} snapshots.")
    except Exception as e:
        print(f"🛑 Error loading file: {e}")
        return

    # Run Scoring
    try:
        # We pass the dataframe to the scoring pipeline
        # It will skip internal aggregation because 'timestamp' is already datetime
        results_df, thresholds = run_anomaly_scoring(df)
        
        if results_df.empty:
            print("⚠️ No results generated. Check if data size > sequence length.")
            return

        # Save Results
        results_df.to_csv(os.path.join(TEST_OUTPUT_DIR, 'slammer_results.csv'), index=False)
        alerts_df = results_df[results_df['severity'] != 'NORMAL']
        
        print(f"✅ Analysis Complete. Found {len(alerts_df)} anomalies.")
        
        # Visualizations
        generate_visualizations(results_df, thresholds, output_dir=TEST_PLOT_DIR)
        
    except Exception as e:
        print(f"🛑 Error during scoring: {e}")

if __name__ == "__main__":
    evaluate_processed_dataset()