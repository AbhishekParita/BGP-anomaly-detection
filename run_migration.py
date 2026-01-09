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

# Read and execute migration
with open('add_ensemble_is_anomaly_column.sql', 'r') as f:
    sql = f.read()
    cur.execute(sql)
    conn.commit()

print("✅ Migration complete - ensemble_is_anomaly column added")

cur.close()
conn.close()
