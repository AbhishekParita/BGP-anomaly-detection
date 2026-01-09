import pandas as pd
from datetime import datetime
import os
import sys

# Add the project root to the path to import local files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the refactored logic
from db_connector import DBConnector
from bmp_generator import MultiRouterSimulator, RouterConfig, PeerConfig
# Assuming you've added the run_anomaly_scoring function to ensemble_bgp_optimized.py
from ensemble_bgp_optimized import run_anomaly_scoring 

# --- PIPELINE CONFIGURATION ---
SIMULATION_HOST = "127.0.0.1" # Host/port are only needed for the original bmp_generator socket logic, 
SIMULATION_PORT = 9999        # but we use them here to initialize the simulator instance.
DATA_POINTS_TO_GENERATE = 1000 # Number of time-window records to generate

def initialize_simulator():
    """Sets up a basic MultiRouterSimulator configuration."""
    
    # 1. Define Peers
    peer1 = PeerConfig(peer_ip="192.168.1.10", peer_as=65001, peer_bgp_id="10.0.0.1")
    peer2 = PeerConfig(peer_ip="192.168.1.20", peer_as=65002, peer_bgp_id="10.0.0.2")

    # 2. Define Router
    router = RouterConfig(
        router_id="172.16.0.1",
        router_ip="172.16.0.1",
        router_as=64500,
        sys_name="Core-Router-1",
        sys_descr="Simulated Core Router",
        peers=[peer1, peer2]
    )

    # 3. Initialize Simulator
    sim = MultiRouterSimulator(SIMULATION_HOST, SIMULATION_PORT)
    sim.add_router(router)
    return sim

def run_pipeline():
    """
    Executes the full pipeline: Generate -> Save -> Process -> Save Results.
    """
    db = DBConnector()
    
    #if not db.conn:
    #    print("Pipeline aborted due to database connection failure.")
    #    return
    connection_error = db.connect() 
    
    if connection_error:
        print("\n==============================================")
        print("🛑 PIPELINE ABORTED DUE TO DATABASE CONNECTION FAILURE.")
        # Print the specific error returned by db_connector
        print(f"PostgreSQL Error: {connection_error}") 
        print("==============================================")
        return

    try:
        # 1. Data Generation and Ingestion
        print("\n=== STEP 1: Data Generation & Ingestion ===")
        sim = initialize_simulator()
        raw_data_list = sim.generate_structured_data(count=DATA_POINTS_TO_GENERATE)
        
        # Insert into database
        db.insert_raw_data(raw_data_list)

        # 2. Data Retrieval for ML
        # We fetch the data we just inserted (or a larger block if needed)
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(hours=24) # Fetch last 24 hours of data
        
        df_for_ml = db.fetch_data_for_ml(start_time, end_time)
        
        if df_for_ml.empty:
            print("No data fetched for ML processing. Pipeline stopped.")
            return

        # 3. ML Processing (Scoring)
        print("\n=== STEP 2: ML Anomaly Scoring ===")
        # Note: run_anomaly_scoring uses the logic from ensemble_bgp_optimized.py
        #results_df = run_anomaly_scoring(df_for_ml)
        results_df, thresholds = run_anomaly_scoring(df_for_ml)

        if results_df.empty:
            print("ML scoring produced no results. Pipeline stopped.")
            return

        # 4. Result Persistence
        print("\n=== STEP 3: Result Persistence ===")
        db.insert_anomaly_results(results_df)

        print("\n==============================================")
        print("✅ PIPELINE COMPLETE: BGP Anomaly Detection Cycle Finished.")
        print(f"Total alerts saved: {len(results_df[results_df['severity'] != 'NORMAL'])}")
        print("==============================================")

    except Exception as e:
        print(f"🛑 CRITICAL PIPELINE FAILURE: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        db.close()

if __name__ == '__main__':
    run_pipeline()