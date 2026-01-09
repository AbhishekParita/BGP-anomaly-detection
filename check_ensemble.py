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

# Check ensemble marking in last 5 minutes
cur.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN ensemble_score > 0.4 THEN 1 ELSE 0 END) as above_threshold,
        SUM(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 ELSE 0 END) as marked_anomaly,
        AVG(ensemble_score) as avg_score
    FROM ml_results 
    WHERE timestamp > NOW() - INTERVAL '5 minutes' 
    AND ensemble_score IS NOT NULL
""")
row = cur.fetchone()
print(f"\n📊 Last 5 minutes:")
print(f"   Total records: {row[0]}")
print(f"   Scores > 0.4 threshold: {row[1]}")
print(f"   Marked as anomaly: {row[2]}")
print(f"   Average score: {row[3]:.3f}" if row[3] else "   Average score: N/A")

# Show sample records
cur.execute("""
    SELECT peer_addr, ensemble_score, ensemble_is_anomaly,
           heuristic_is_anomaly, lstm_is_anomaly, if_is_anomaly
    FROM ml_results 
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    AND ensemble_score IS NOT NULL
    ORDER BY ensemble_score DESC
    LIMIT 5
""")
print(f"\n🔍 Top 5 highest ensemble scores:")
for row in cur.fetchall():
    print(f"   Peer: {row[0]}, Score: {row[1]:.3f}, Ensemble: {row[2]}, H: {row[3]}, L: {row[4]}, IF: {row[5]}")

cur.close()
conn.close()
