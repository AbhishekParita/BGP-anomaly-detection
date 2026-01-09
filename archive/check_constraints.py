"""Check ml_results table constraints"""
import psycopg2

conn = psycopg2.connect(
    dbname='bgp_monitor',
    user='postgres',
    password='anand',
    host='localhost'
)
cur = conn.cursor()

print("="*70)
print("ML_RESULTS TABLE CONSTRAINTS")
print("="*70)

# Check constraints
cur.execute("""
    SELECT constraint_name, constraint_type 
    FROM information_schema.table_constraints 
    WHERE table_name = 'ml_results';
""")

print("\nConstraints:")
for row in cur.fetchall():
    print(f"  {row[0]:40s} {row[1]}")

# Check unique constraints details
cur.execute("""
    SELECT kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
    WHERE tc.table_name = 'ml_results' 
        AND tc.constraint_type = 'UNIQUE';
""")

unique_cols = cur.fetchall()
if unique_cols:
    print("\nUnique constraint on columns:")
    for col in unique_cols:
        print(f"  - {col[0]}")

print("\n" + "="*70)
print("PROBLEM DIAGNOSIS:")
print("="*70)

# Check if multiple detectors are trying to write for same feature
cur.execute("""
    SELECT feature_id, COUNT(*) as detector_count,
           COUNT(CASE WHEN heuristic_score IS NOT NULL THEN 1 END) as has_heuristic,
           COUNT(CASE WHEN lstm_anomaly_score IS NOT NULL THEN 1 END) as has_lstm,
           COUNT(CASE WHEN if_anomaly_score IS NOT NULL THEN 1 END) as has_if
    FROM ml_results
    GROUP BY feature_id
    HAVING COUNT(*) > 1
    LIMIT 5;
""")

multi_detector = cur.fetchall()
if multi_detector:
    print("\n✅ Multiple records per feature (GOOD - each detector writes separately):")
    for row in multi_detector:
        print(f"  Feature {row[0]}: {row[1]} records (H:{row[2]}, L:{row[3]}, IF:{row[4]})")
else:
    print("\n⚠️ Each feature has only 1 record (detectors might be conflicting)")

# Check if LSTM/IF are being blocked
cur.execute("""
    SELECT 
        COUNT(CASE WHEN heuristic_score IS NOT NULL THEN 1 END) as heuristic_count,
        COUNT(CASE WHEN lstm_anomaly_score IS NOT NULL THEN 1 END) as lstm_count,
        COUNT(CASE WHEN if_anomaly_score IS NOT NULL THEN 1 END) as if_count
    FROM ml_results;
""")

counts = cur.fetchone()
print(f"\n📊 Detector coverage:")
print(f"  Heuristic scores: {counts[0]:,}")
print(f"  LSTM scores: {counts[1]:,}")
print(f"  IF scores: {counts[2]:,}")

if counts[0] > 0 and counts[1] == 0 and counts[2] == 0:
    print("\n❌ PROBLEM: Only heuristic is writing!")
    print("   → LSTM and IF detectors are NOT inserting their results")
    print("   → Check detector logs for errors")

conn.close()
