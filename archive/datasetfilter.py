import pandas as pd
import numpy as np

# --- CONFIGURATION ---
FILE_NAME = 'E:\Advantal_models\lstm model/raw_data\code_red.csv'  # Can be 'nimda.csv' or 'code_red.csv'
OUTPUT_NAME = 'processed_distributed_test_RED.csv'

# Historical Start Dates (The "Normal" day)
EVENT_DATES = {
    'slammer': '2003-01-25',
    'nimda': '2001-09-18',
    'code_red': '2001-07-19'
}

def preprocess_bgp_distributed(file_path):
    print(f"--- Processing {file_path} ---")
    df = pd.read_csv(file_path)
    
    # 1. IDENTIFY BASE EVENT DATE
    event_key = next((k for k in EVENT_DATES if k in file_path.lower()), 'slammer')
    base_date_str = EVENT_DATES[event_key]
    base_date = pd.to_datetime(base_date_str)
    
    # 2. CREATE INITIAL TIMESTAMP (Normal data stays on base_date)
    print("Creating timestamps...")
    df['timestamp'] = base_date + \
                      pd.to_timedelta(df['Hour'], unit='h') + \
                      pd.to_timedelta(df['Minutes'], unit='m') + \
                      pd.to_timedelta(df['Seconds'], unit='s')

    # --- NEW REQUIREMENT: Distribute Anomalies across 3 different dates ---
    anomaly_indices = df[df['Label'] == 1].index.tolist()
    n = len(anomaly_indices)
    
    if n > 0:
        print(f"Distributing {n} anomaly rows across D-2, D-1, and D+1...")
        # Split anomaly indices into 3 parts
        p1 = anomaly_indices[:n//3]
        p2 = anomaly_indices[n//3 : 2*n//3]
        p3 = anomaly_indices[2*n//3:]
        
        # Calculate target dates
        date_m2 = base_date - pd.Timedelta(days=2)
        date_m1 = base_date - pd.Timedelta(days=1)
        date_p1 = base_date + pd.Timedelta(days=1)
        
        # Re-assign dates for anomalies (keeping the original Hour/Min/Sec for spread)
        df.loc[p1, 'timestamp'] = df.loc[p1, 'timestamp'].apply(lambda x: x.replace(year=date_m2.year, month=date_m2.month, day=date_m2.day))
        df.loc[p2, 'timestamp'] = df.loc[p2, 'timestamp'].apply(lambda x: x.replace(year=date_m1.year, month=date_m1.month, day=date_m1.day))
        df.loc[p3, 'timestamp'] = df.loc[p3, 'timestamp'].apply(lambda x: x.replace(year=date_p1.year, month=date_p1.month, day=date_p1.day))

    # 3. MAP TO THE 9 FEATURES FOR ENSEMBLE MODEL
    print("Mapping features...")
    processed = pd.DataFrame()
    processed['timestamp'] = df['timestamp']
    processed['announcements'] = df['Number of announcements']
    processed['withdrawals'] = df['Number of withdrawals']
    processed['total_updates'] = processed['announcements'] + processed['withdrawals']
    processed['withdrawal_ratio'] = processed['withdrawals'] / processed.apply(lambda x: max(x['total_updates'], 1), axis=1)
    processed['flap_count'] = df['Average edit distance'] 
    processed['path_length'] = df['Average AS-path length']
    processed['unique_peers'] = 1 
    processed['message_rate'] = processed['total_updates'] 
    processed['session_resets'] = 0 
    
    # 4. HANDLE LABELS
    processed['anomaly_type'] = df['Label'].apply(lambda x: 'attack' if x == 1 else 'normal')
    
    # 5. FINAL TOUCHES: Sort and Save
    # We sort by timestamp so the graph shows the timeline correctly (D-2 -> D-1 -> D -> D+1)
    processed = processed.sort_values('timestamp')
    
    # Save the file
    processed.to_csv(OUTPUT_NAME, index=False)
    print(f"✅ Success! Processed data saved as: {OUTPUT_NAME}")
    print(f"The time series now spans from {processed['timestamp'].min()} to {processed['timestamp'].max()}")

if __name__ == "__main__":
    preprocess_bgp_distributed(FILE_NAME)
