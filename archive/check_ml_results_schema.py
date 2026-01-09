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

print("Checking ml_results table structure...")

# Check if ensemble_is_anomaly column exists
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'ml_results'
    ORDER BY ordinal_position;
""")

columns = cur.fetchall()
print("\nColumns in ml_results table:")
print("-" * 50)
for col, dtype in columns:
    print(f"{col:30s} {dtype}")

# Check for ensemble_is_anomaly specifically
has_ensemble_is_anomaly = any(col[0] == 'ensemble_is_anomaly' for col in columns)
print("\n" + "=" * 50)
if has_ensemble_is_anomaly:
    print("✅ ensemble_is_anomaly column EXISTS")
else:
    print("❌ ensemble_is_anomaly column MISSING!")
    print("\nThis column needs to be added to the table.")

conn.close()
