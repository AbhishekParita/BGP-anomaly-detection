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

print("=" * 60)
print("CHECKING RECENT ML_RESULTS RECORDS")
print("=" * 60)

# Check recent records
cur.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN ensemble_score IS NOT NULL THEN 1 ELSE 0 END) as has_score,
        SUM(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 ELSE 0 END) as marked_anomaly
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
""")

row = cur.fetchone()
print(f"\n📊 Records (last 10 minutes):")
print(f"   Total records: {row[0]}")
print(f"   Has ensemble_score: {row[1]}")
print(f"   Marked as anomaly: {row[2]}")

# Check a sample of recent records
cur.execute("""
    SELECT 
        id, timestamp, peer_addr,
        ensemble_score, ensemble_is_anomaly,
        heuristic_is_anomaly, lstm_is_anomaly, if_is_anomaly
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    ORDER BY timestamp DESC
    LIMIT 10
""")

print(f"\n📋 Sample of 10 most recent records:")
print("-" * 60)
print(f"{'ID':<8} {'Score':<8} {'Ensemble':<10} {'H':<4} {'L':<4} {'IF':<4}")
print("-" * 60)

for row in cur.fetchall():
    score = f"{row[3]:.3f}" if row[3] is not None else "NULL"
    ens = "TRUE" if row[4] else "FALSE"
    h = "T" if row[5] else "F"
    l = "T" if row[6] else "F"
    i = "T" if row[7] else "F"
    print(f"{row[0]:<8} {score:<8} {ens:<10} {h:<4} {l:<4} {i:<4}")

print("=" * 60)

conn.close()
