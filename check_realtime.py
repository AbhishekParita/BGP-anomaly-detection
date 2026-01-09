"""Check why data is not real-time"""
import psycopg2
from datetime import datetime

conn = psycopg2.connect(
    dbname='bgp_monitor',
    user='postgres',
    password='anand',
    host='localhost'
)
cur = conn.cursor()

print("="*70)
print("REAL-TIME DATA CHECK")
print("="*70)

# Check latest raw data
cur.execute("SELECT MAX(timestamp) as latest, COUNT(*) FROM raw_bgp_data;")
raw = cur.fetchone()
raw_age = (datetime.now() - raw[0]).total_seconds() if raw[0] else None
print(f"\n[1] Raw BGP Data:")
print(f"  Latest: {raw[0]}")
print(f"  Age: {raw_age:.0f} seconds ({raw_age/60:.1f} minutes)" if raw_age else "  Age: N/A")
print(f"  Total: {raw[1]:,}")

# Check latest features
cur.execute("SELECT MAX(timestamp) as latest, COUNT(*) FROM features;")
feat = cur.fetchone()
feat_age = (datetime.now() - feat[0]).total_seconds() if feat[0] else None
print(f"\n[2] Feature Processing:")
print(f"  Latest: {feat[0]}")
print(f"  Age: {feat_age:.0f} seconds ({feat_age/60:.1f} minutes)" if feat_age else "  Age: N/A")
print(f"  Total: {feat[1]:,}")

# Check latest ML results
cur.execute("SELECT MAX(timestamp) as latest, COUNT(*) FROM ml_results;")
ml = cur.fetchone()
ml_age = (datetime.now() - ml[0]).total_seconds() if ml[0] else None
print(f"\n[3] ML Detection:")
print(f"  Latest: {ml[0]}")
print(f"  Age: {ml_age:.0f} seconds ({ml_age/60:.1f} minutes)" if ml_age else "  Age: N/A")
print(f"  Total: {ml[1]:,}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if raw_age and raw_age > 300:  # 5 minutes
    print("❌ RIS Live Collector is STOPPED or SLOW")
    print("   → Data is over 5 minutes old")
    print("   → Restart: python services/ris_live_collector.py")
elif feat_age and feat_age > 120:  # 2 minutes
    print("❌ Feature Aggregator is STOPPED or SLOW")
    print("   → Raw data is fresh but features are old")
    print("   → Check feature_aggregator service")
elif ml_age and ml_age > 120:  # 2 minutes
    print("❌ ML Detectors are STOPPED or SLOW")
    print("   → Features are fresh but ML results are old")
    print("   → Check detector services")
else:
    print("✅ All services running with fresh data!")

conn.close()
