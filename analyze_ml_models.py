import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

print("=" * 70)
print("ML MODELS DETECTION ANALYSIS")
print("=" * 70)

# Check which models are actually detecting anomalies
cur.execute("""
    SELECT 
        COUNT(*) as total_records,
        
        -- Heuristic
        SUM(CASE WHEN heuristic_score IS NOT NULL THEN 1 ELSE 0 END) as heuristic_has_score,
        SUM(CASE WHEN heuristic_is_anomaly = TRUE THEN 1 ELSE 0 END) as heuristic_detections,
        AVG(CASE WHEN heuristic_score IS NOT NULL THEN heuristic_score ELSE NULL END) as heuristic_avg_score,
        
        -- LSTM
        SUM(CASE WHEN lstm_anomaly_score IS NOT NULL THEN 1 ELSE 0 END) as lstm_has_score,
        SUM(CASE WHEN lstm_is_anomaly = TRUE THEN 1 ELSE 0 END) as lstm_detections,
        AVG(CASE WHEN lstm_anomaly_score IS NOT NULL THEN lstm_anomaly_score ELSE NULL END) as lstm_avg_score,
        
        -- Isolation Forest
        SUM(CASE WHEN if_anomaly_score IS NOT NULL THEN 1 ELSE 0 END) as if_has_score,
        SUM(CASE WHEN if_is_anomaly = TRUE THEN 1 ELSE 0 END) as if_detections,
        AVG(CASE WHEN if_anomaly_score IS NOT NULL THEN if_anomaly_score ELSE NULL END) as if_avg_score,
        
        -- Ensemble
        SUM(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 ELSE 0 END) as ensemble_detections,
        AVG(ensemble_score) as ensemble_avg_score
        
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '30 minutes'
""")

row = cur.fetchone()

print(f"\n📊 Analysis (last 30 minutes):")
print(f"   Total records: {row[0]}")
print()

# Heuristic
heuristic_avg = f"{row[3]:.3f}" if row[3] else "0.000"
print(f"🔍 HEURISTIC Detector:")
print(f"   Records with scores: {row[1]}")
print(f"   Anomalies detected: {row[2]}")
print(f"   Average score: {heuristic_avg}")
print(f"   Detection rate: {(row[2]/row[0]*100):.1f}%" if row[0] > 0 else "   Detection rate: 0.0%")
print()

# LSTM
lstm_avg = f"{row[6]:.3f}" if row[6] else "0.000"
print(f"🧠 LSTM Detector:")
print(f"   Records with scores: {row[4]}")
print(f"   Anomalies detected: {row[5]}")
print(f"   Average score: {lstm_avg}")
print(f"   Detection rate: {(row[5]/row[0]*100):.1f}%" if row[0] > 0 else "   Detection rate: 0.0%")
if row[5] == 0 and row[4] > 0:
    print(f"   ⚠️  WARNING: Has scores but detecting 0 anomalies!")
print()

# Isolation Forest
if_avg = f"{row[9]:.3f}" if row[9] else "0.000"
print(f"🌲 ISOLATION FOREST Detector:")
print(f"   Records with scores: {row[7]}")
print(f"   Anomalies detected: {row[8]}")
print(f"   Average score: {if_avg}")
print(f"   Detection rate: {(row[8]/row[0]*100):.1f}%" if row[0] > 0 else "   Detection rate: 0.0%")
if row[8] == 0 and row[7] > 0:
    print(f"   ⚠️  WARNING: Has scores but detecting 0 anomalies!")
if row[7] == 0:
    print(f"   ❌ ERROR: Model not generating any scores!")
print()

# Ensemble
ensemble_avg = f"{row[11]:.3f}" if row[11] else "0.000"
print(f"⚡ ENSEMBLE:")
print(f"   Anomalies detected: {row[10]}")
print(f"   Average score: {ensemble_avg}")
print(f"   Detection rate: {(row[10]/row[0]*100):.1f}%" if row[0] > 0 else "   Detection rate: 0.0%")
print()

print("=" * 70)
print("SUMMARY:")
if row[5] == 0 and row[4] > 0:
    print("❌ LSTM is broken - has scores but not detecting anomalies")
if row[8] == 0 and row[7] > 0:
    print("❌ Isolation Forest is broken - has scores but not detecting anomalies")
if row[7] == 0:
    print("❌ Isolation Forest model is NOT LOADED or NOT RUNNING")
if row[10]/row[0] > 0.5:
    print(f"⚠️  WARNING: {(row[10]/row[0]*100):.1f}% anomaly rate is VERY HIGH!")
    print("   Possible causes:")
    print("   - Threshold too low (current: 0.4)")
    print("   - Only Heuristic detector working")
    print("   - Training data mismatch with live data")
print("=" * 70)

conn.close()
