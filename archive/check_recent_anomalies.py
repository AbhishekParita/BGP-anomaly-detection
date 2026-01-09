"""Check recent anomaly detections"""
import psycopg2

conn = psycopg2.connect(
    dbname='bgp_monitor',
    user='postgres',
    password='anand',
    host='localhost'
)
cur = conn.cursor()

print("="*70)
print("RECENT ANOMALY DETECTION STATUS")
print("="*70)

# Check recent ml_results with scores
print("\n[1] Recent ML Results (last 10 minutes):")
cur.execute("""
    SELECT timestamp, peer_addr,
           heuristic_is_anomaly, lstm_is_anomaly, if_is_anomaly,
           heuristic_score, lstm_anomaly_score, if_anomaly_score,
           ensemble_score, ensemble_is_anomaly
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
    ORDER BY timestamp DESC
    LIMIT 10;
""")

recent = cur.fetchall()
if recent:
    print(f"  Found {len(recent)} recent detections")
    for r in recent[:5]:
        ts, peer, h_anom, l_anom, if_anom, h_score, l_score, if_score, ens, ens_anom = r
        print(f"\n  {ts} | {peer[:20]}")
        print(f"    H: {h_anom} ({h_score:.3f if h_score else 'None'})")
        print(f"    L: {l_anom} ({l_score:.3f if l_score else 'None'})")
        print(f"    IF: {if_anom} ({if_score:.3f if if_score else 'None'})")
        print(f"    Ensemble: {ens_anom} ({ens:.3f if ens else 0})")
else:
    print("  ⚠️ No recent detections!")

# Count anomalies in last hour
print("\n[2] Anomaly counts (last hour):")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN heuristic_is_anomaly = TRUE THEN 1 END) as h_anomalies,
        COUNT(CASE WHEN lstm_is_anomaly = TRUE THEN 1 END) as l_anomalies,
        COUNT(CASE WHEN if_is_anomaly = TRUE THEN 1 END) as if_anomalies,
        COUNT(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 END) as ens_anomalies
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '1 hour';
""")

counts = cur.fetchone()
print(f"  Total detections: {counts[0]}")
print(f"  Heuristic anomalies: {counts[1]}")
print(f"  LSTM anomalies: {counts[2]}")
print(f"  IF anomalies: {counts[3]}")
print(f"  Ensemble anomalies: {counts[4]}")

# Check alert creation
print("\n[3] Alert creation status:")
cur.execute("""
    SELECT COUNT(*), MAX(timestamp) as latest
    FROM alerts
    WHERE timestamp > NOW() - INTERVAL '1 hour';
""")
alert_count, latest_alert = cur.fetchone()
print(f"  Alerts in last hour: {alert_count}")
print(f"  Latest alert: {latest_alert if latest_alert else 'None'}")

# Check correlation engine input
print("\n[4] Checking correlation engine input (ensemble_results):")
cur.execute("""
    SELECT COUNT(*), MAX(timestamp) as latest
    FROM ensemble_results
    WHERE timestamp > NOW() - INTERVAL '10 minutes';
""")
ens_res = cur.fetchone()
print(f"  Ensemble results (last 10 min): {ens_res[0]}")
print(f"  Latest: {ens_res[1] if ens_res[1] else 'None'}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if counts[4] == 0:
    print("⚠️ NO ANOMALIES detected in last hour")
    print("   → All BGP traffic is NORMAL (no attacks/anomalies)")
    print("   → Dashboard won't update because there are no new alerts")
    print("\n💡 This is actually GOOD - your network is healthy!")
    print("   To test anomaly detection:")
    print("   1. Run: python test_new_data.py (with test anomaly data)")
    print("   2. Wait for test data with actual anomalies")
elif alert_count == 0 and counts[4] > 0:
    print(f"❌ {counts[4]} anomalies detected but NO ALERTS created!")
    print("   → Correlation Engine is NOT working")
    print("   → Check correlation_engine.log")
else:
    print(f"✅ System working! {counts[4]} anomalies → {alert_count} alerts")

conn.close()
