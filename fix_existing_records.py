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
print("FIXING INCORRECTLY MARKED RECORDS")
print("=" * 60)

# Find records with score > 0.4 but marked as FALSE
cur.execute("""
    SELECT COUNT(*)
    FROM ml_results
    WHERE ensemble_score > 0.4
    AND ensemble_is_anomaly = FALSE
""")
count = cur.fetchone()[0]
print(f"\n📊 Found {count} records incorrectly marked as FALSE")

if count > 0:
    print(f"🔧 Fixing {count} records...")
    
    # Fix them
    cur.execute("""
        UPDATE ml_results
        SET ensemble_is_anomaly = TRUE
        WHERE ensemble_score > 0.4
        AND ensemble_is_anomaly = FALSE
    """)
    
    rows_updated = cur.rowcount
    conn.commit()
    
    print(f"✅ Successfully updated {rows_updated} records to TRUE")
    
    # Verify
    cur.execute("""
        SELECT COUNT(*)
        FROM ml_results
        WHERE ensemble_score > 0.4
        AND ensemble_is_anomaly = TRUE
    """)
    correct_count = cur.fetchone()[0]
    print(f"✅ Verified: {correct_count} records now correctly marked as TRUE")
else:
    print("✅ No records need fixing")

print("\n" + "=" * 60)

conn.close()
