import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

print("=" * 60)
print("BGP ANOMALY DETECTION - REAL-TIME STATUS")
print("=" * 60)

# Check total alerts
cur.execute("SELECT COUNT(*) FROM alerts")
total_alerts = cur.fetchone()[0]
print(f"✅ Total alerts: {total_alerts}")

# Check new alerts
cur.execute("SELECT COUNT(*) FROM alerts WHERE timestamp > NOW() - INTERVAL '10 minutes'")
new_alerts = cur.fetchone()[0]
print(f"📊 New alerts (last 10 min): {new_alerts}")

# Check total anomalies detected
cur.execute("SELECT COUNT(*) FROM ml_results WHERE ensemble_is_anomaly = TRUE")
total_anomalies = cur.fetchone()[0]
print(f"🔍 Total anomalies detected: {total_anomalies}")

# Check unprocessed anomalies (no alert yet)
cur.execute("""
    SELECT COUNT(*) FROM ml_results m
    WHERE m.ensemble_is_anomaly = TRUE
    AND NOT EXISTS (
        SELECT 1 FROM alerts a WHERE a.ml_result_id = m.id
    )
""")
unprocessed = cur.fetchone()[0]
print(f"⏳ Unprocessed anomalies: {unprocessed}")

# Check recent detections
cur.execute("SELECT COUNT(*) FROM ml_results WHERE timestamp > NOW() - INTERVAL '10 minutes'")
recent_detections = cur.fetchone()[0]
print(f"📈 Recent detections (last 10 min): {recent_detections}")

# Check latest alert timestamp
cur.execute("SELECT MAX(timestamp) FROM alerts")
latest_alert = cur.fetchone()[0]
print(f"🕐 Latest alert time: {latest_alert}")

print("=" * 60)

conn.close()
