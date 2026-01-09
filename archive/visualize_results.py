import pandas as pd
import matplotlib.pyplot as plt
from db_connector import DBConnector

# --- CONFIGURATION ---
# Assumes your .env file is correctly set up for 'mydb'
DB_NAME = 'mydb' 

def visualize_anomaly_results():
    """Fetches raw data and anomaly alerts and plots them."""
    db = DBConnector()
    
    # 1. Call connect(), which prints the success message and sets db.conn
    db.connect() 
    
    # 2. CRITICAL CHANGE: Check if the connection object (db.conn) exists.
    #    This is more reliable than checking the boolean return value.
    if db.conn is None:
        print("Failed to establish database connection. Check .env file and PostgreSQL service.")
        return

    try:
        # --- 1. Fetch Raw Data (We need a time range, e.g., last 2 days) ---
        print("Fetching raw BGP data...")
        raw_sql = """
            SELECT timestamp, announcements, withdrawals, total_updates, withdrawal_ratio
            FROM public.raw_bgp_data
            ORDER BY timestamp;
        """
        raw_df = pd.read_sql(raw_sql, db.conn)
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
        print(f"Fetched {len(raw_df)} raw records.")

        # --- 2. Fetch Anomaly Alerts ---
        print("Fetching anomaly alerts...")
        alert_sql = """
            SELECT timestamp, severity
            FROM public.anomaly_alerts
            WHERE severity != 'NORMAL'
            ORDER BY timestamp;
        """
        alert_df = pd.read_sql(alert_sql, db.conn)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        print(f"Fetched {len(alert_df)} alert records.")

    except Exception as e:
        print(f"Error fetching data for visualization: {e}")
        db.close()
        return
    finally:
        db.close()

    if raw_df.empty:
        print("No raw data to plot.")
        return

    # --- 3. PLOT VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the primary time-series metric (Total Updates)
    ax.plot(raw_df['timestamp'], raw_df['total_updates'], label='Total Updates', color='skyblue', linewidth=1.5)

    # Merge alerts with raw data to get the metric value at the time of the alert
    merged_df = pd.merge(alert_df, raw_df, on='timestamp', how='left')
    
    # Filter for Critical and High Alerts
    critical_alerts = merged_df[merged_df['severity'] == 'CRITICAL']
    high_alerts = merged_df[merged_df['severity'] == 'HIGH']
    
    # Overlay the critical anomalies
    ax.scatter(critical_alerts['timestamp'], critical_alerts['total_updates'], 
               color='red', marker='X', s=100, label='CRITICAL Anomaly')
    
    # Overlay the high anomalies
    ax.scatter(high_alerts['timestamp'], high_alerts['total_updates'], 
               color='orange', marker='o', s=60, label='HIGH Anomaly')

    # Formatting
    ax.set_title('BGP Total Updates Time Series with Detected Anomalies', fontsize=16)
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Total Updates Count', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_anomaly_results()