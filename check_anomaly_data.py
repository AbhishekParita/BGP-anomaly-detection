"""
Check current anomaly detection data in database.
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

db_config = {
    'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Check total records
    cursor.execute("SELECT COUNT(*) FROM ml_results;")
    total = cursor.fetchone()[0]
    print(f"Total ml_results records: {total}")
    
    # Check ensemble anomalies
    cursor.execute("SELECT COUNT(*) FROM ml_results WHERE ensemble_is_anomaly = TRUE;")
    ensemble_anomalies = cursor.fetchone()[0]
    print(f"Ensemble anomalies (TRUE): {ensemble_anomalies}")
    
    # Check individual detector anomalies
    cursor.execute("SELECT COUNT(*) FROM ml_results WHERE heuristic_is_anomaly = TRUE;")
    heuristic_anomalies = cursor.fetchone()[0]
    print(f"Heuristic anomalies: {heuristic_anomalies}")
    
    cursor.execute("SELECT COUNT(*) FROM ml_results WHERE lstm_is_anomaly = TRUE;")
    lstm_anomalies = cursor.fetchone()[0]
    print(f"LSTM anomalies: {lstm_anomalies}")
    
    cursor.execute("SELECT COUNT(*) FROM ml_results WHERE if_is_anomaly = TRUE;")
    if_anomalies = cursor.fetchone()[0]
    print(f"Isolation Forest anomalies: {if_anomalies}")
    
    # Check ensemble score distribution
    cursor.execute("""
        SELECT 
            MIN(ensemble_score) as min_score,
            AVG(ensemble_score) as avg_score,
            MAX(ensemble_score) as max_score
        FROM ml_results
        WHERE ensemble_score IS NOT NULL;
    """)
    scores = cursor.fetchone()
    print(f"\nEnsemble score distribution:")
    print(f"  Min: {scores[0]:.4f}")
    print(f"  Avg: {scores[1]:.4f}")
    print(f"  Max: {scores[2]:.4f}")
    
    # Show top 5 highest scores
    cursor.execute("""
        SELECT peer_addr, ensemble_score, ensemble_is_anomaly,
               heuristic_is_anomaly, lstm_is_anomaly, if_is_anomaly
        FROM ml_results
        WHERE ensemble_score IS NOT NULL
        ORDER BY ensemble_score DESC
        LIMIT 5;
    """)
    print(f"\nTop 5 highest ensemble scores:")
    for row in cursor.fetchall():
        print(f"  Peer: {row[0]}, Score: {row[1]:.4f}, Ensemble: {row[2]}, H: {row[3]}, L: {row[4]}, IF: {row[5]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
