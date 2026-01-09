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
print("COMPLETE PIPELINE DIAGNOSIS")
print("=" * 70)

# STEP 1: Check raw BGP data collection
print("\n[STEP 1] RIS Live Collector → raw_bgp_data")
cur.execute("SELECT COUNT(*), MAX(timestamp) FROM raw_bgp_data WHERE timestamp > NOW() - INTERVAL '10 minutes'")
row = cur.fetchone()
print(f"  ✓ Records collected (last 10 min): {row[0]}")
print(f"  ✓ Latest timestamp: {row[1]}")

# STEP 2: Check feature aggregation
print("\n[STEP 2] Feature Aggregator → features")
cur.execute("SELECT COUNT(*), MAX(timestamp) FROM features WHERE timestamp > NOW() - INTERVAL '10 minutes'")
row = cur.fetchone()
print(f"  ✓ Features aggregated (last 10 min): {row[0]}")
print(f"  ✓ Latest timestamp: {row[1]}")

# STEP 3: Check detector outputs
print("\n[STEP 3] Detectors → ml_results")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN heuristic_is_anomaly = TRUE THEN 1 ELSE 0 END) as heuristic_true,
        SUM(CASE WHEN lstm_is_anomaly = TRUE THEN 1 ELSE 0 END) as lstm_true,
        SUM(CASE WHEN if_is_anomaly = TRUE THEN 1 ELSE 0 END) as if_true
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
""")
row = cur.fetchone()
print(f"  ✓ ML results created (last 10 min): {row[0]}")
print(f"  ✓ Heuristic marked TRUE: {row[1]}")
print(f"  ✓ LSTM marked TRUE: {row[2]}")
print(f"  ✓ IF marked TRUE: {row[3]}")

# STEP 4: Check ensemble scores
print("\n[STEP 4] Ensemble Coordinator → ensemble_score")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        AVG(ensemble_score) as avg_score,
        MIN(ensemble_score) as min_score,
        MAX(ensemble_score) as max_score,
        SUM(CASE WHEN ensemble_score > 0.4 THEN 1 ELSE 0 END) as above_threshold
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
    AND ensemble_score IS NOT NULL
""")
row = cur.fetchone()
print(f"  ✓ Records with ensemble_score: {row[0]}")
avg_score = f"{row[1]:.3f}" if row[1] else "0.000"
min_score = f"{row[2]:.3f}" if row[2] else "0.000"
max_score = f"{row[3]:.3f}" if row[3] else "0.000"
print(f"  ✓ Average score: {avg_score}")
print(f"  ✓ Score range: {min_score} - {max_score}")
print(f"  ✓ Scores > 0.4 threshold: {row[4]}")

# STEP 5: Check ensemble_is_anomaly flag
print("\n[STEP 5] Ensemble Coordinator → ensemble_is_anomaly")
cur.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 ELSE 0 END) as marked_true,
        SUM(CASE WHEN ensemble_is_anomaly = FALSE THEN 1 ELSE 0 END) as marked_false,
        SUM(CASE WHEN ensemble_is_anomaly IS NULL THEN 1 ELSE 0 END) as marked_null
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
    AND ensemble_score IS NOT NULL
""")
row = cur.fetchone()
print(f"  ✓ Total records: {row[0]}")
print(f"  ❌ ensemble_is_anomaly = TRUE: {row[1]}")
print(f"  ⚠️  ensemble_is_anomaly = FALSE: {row[2]}")
print(f"  ⚠️  ensemble_is_anomaly = NULL: {row[3]}")

# STEP 6: Detailed sample analysis
print("\n[STEP 6] Sample Records Analysis")
cur.execute("""
    SELECT id, ensemble_score, ensemble_is_anomaly, heuristic_is_anomaly
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    AND ensemble_score > 0.4
    ORDER BY ensemble_score DESC
    LIMIT 5
""")
print(f"  Top 5 records with score > 0.4:")
for row in cur.fetchall():
    print(f"    ID={row[0]}, Score={row[1]:.3f}, ensemble_is_anomaly={row[2]}, heuristic={row[3]}")

# STEP 7: Check correlation engine
print("\n[STEP 7] Correlation Engine → alerts")
cur.execute("SELECT COUNT(*) FROM alerts WHERE timestamp > NOW() - INTERVAL '10 minutes'")
alerts = cur.fetchone()[0]
print(f"  ✓ Alerts created (last 10 min): {alerts}")

# STEP 8: Test manual UPDATE
print("\n[STEP 8] Testing Manual UPDATE")
cur.execute("""
    SELECT id, ensemble_score, ensemble_is_anomaly
    FROM ml_results
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    AND ensemble_score > 0.4
    LIMIT 1
""")
test_row = cur.fetchone()
if test_row:
    test_id, test_score, current_value = test_row
    print(f"  Testing on ID={test_id}, score={test_score:.3f}, current={current_value}")
    
    # Try to update it
    cur.execute("""
        UPDATE ml_results 
        SET ensemble_is_anomaly = TRUE 
        WHERE id = %s
    """, (test_id,))
    rows_affected = cur.rowcount
    conn.commit()
    
    print(f"  UPDATE affected {rows_affected} rows")
    
    # Check if it worked
    cur.execute("SELECT ensemble_is_anomaly FROM ml_results WHERE id = %s", (test_id,))
    new_value = cur.fetchone()[0]
    print(f"  After UPDATE: ensemble_is_anomaly = {new_value}")
    
    if new_value == True:
        print(f"  ✅ Manual UPDATE WORKS! Problem is in Ensemble Coordinator code.")
    else:
        print(f"  ❌ Manual UPDATE FAILED! Database/schema issue!")
else:
    print(f"  ⚠️  No records with score > 0.4 found for testing")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

conn.close()
