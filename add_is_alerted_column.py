import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

print("Adding is_alerted column to ml_results table...")

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

try:
    # Add is_alerted column
    cur.execute("""
        ALTER TABLE ml_results 
        ADD COLUMN IF NOT EXISTS is_alerted BOOLEAN DEFAULT FALSE;
    """)
    
    # Create index for faster queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ml_results_is_alerted 
        ON ml_results(is_alerted, timestamp DESC) 
        WHERE is_alerted = FALSE;
    """)
    
    conn.commit()
    print("✅ Successfully added is_alerted column and index")
    
    # Check current state
    cur.execute("SELECT COUNT(*) FROM ml_results WHERE is_alerted = FALSE")
    unalerted = cur.fetchone()[0]
    print(f"📊 Unalerted anomalies ready for processing: {unalerted}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
finally:
    conn.close()
