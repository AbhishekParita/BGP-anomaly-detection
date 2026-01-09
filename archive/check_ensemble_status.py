"""Check if Ensemble Coordinator is working"""
import psycopg2

conn = psycopg2.connect(
    dbname='bgp_monitor',
    user='postgres',
    password='anand',
    host='localhost'
)
cur = conn.cursor()

print("="*70)
print("ENSEMBLE COORDINATOR STATUS CHECK")
print("="*70)

# 1. Check total ml_results
cur.execute("SELECT COUNT(*) FROM ml_results;")
total = cur.fetchone()[0]
print(f"\n✅ Total ML results: {total:,}")

# 2. Check records with ensemble_score
cur.execute("SELECT COUNT(*) FROM ml_results WHERE ensemble_score IS NOT NULL;")
with_ensemble = cur.fetchone()[0]
print(f"✅ With ensemble_score: {with_ensemble:,}")

# 3. Check records WITHOUT ensemble_score (pending processing)
cur.execute("SELECT COUNT(*) FROM ml_results WHERE ensemble_score IS NULL;")
pending = cur.fetchone()[0]
print(f"⚠️  Pending (no ensemble_score): {pending:,}")

# 4. Check if any recent records have ensemble scores
cur.execute("""
    SELECT COUNT(*) FROM ml_results 
    WHERE ensemble_score IS NOT NULL 
    AND timestamp > NOW() - INTERVAL '5 minutes';
""")
recent_ensemble = cur.fetchone()[0]
print(f"✅ Recent ensemble (last 5 min): {recent_ensemble:,}")

# 5. Show sample records with scores
print("\n" + "="*70)
print("SAMPLE ML RESULTS (Latest 3):")
print("="*70)
cur.execute("""
    SELECT timestamp, peer_addr,
           heuristic_score, lstm_anomaly_score, if_anomaly_score,
           ensemble_score, ensemble_is_anomaly
    FROM ml_results 
    ORDER BY id DESC 
    LIMIT 3;
""")

for row in cur.fetchall():
    ts, peer, h_score, lstm_score, if_score, ens_score, ens_anom = row
    print(f"\nTime: {ts}")
    print(f"Peer: {peer}")
    print(f"  Heuristic: {h_score}")
    print(f"  LSTM: {lstm_score}")
    print(f"  IF: {if_score}")
    print(f"  Ensemble: {ens_score} | Anomaly: {ens_anom}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if pending > 0 and recent_ensemble == 0:
    print(f"❌ PROBLEM: {pending:,} records waiting for ensemble processing!")
    print("   → Ensemble Coordinator is NOT running or has errors")
    print("   → Check ensemble_coordinator.log for errors")
elif pending == 0:
    print("✅ All records processed! Ensemble working perfectly.")
else:
    print(f"✅ Ensemble Coordinator is processing ({recent_ensemble} recent)")

conn.close()
