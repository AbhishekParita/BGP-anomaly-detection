"""
Add missing ensemble_is_anomaly column to ml_results table.
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Database configuration
db_config = {
    'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Add missing column
    cursor.execute("""
        ALTER TABLE ml_results 
        ADD COLUMN IF NOT EXISTS ensemble_is_anomaly BOOLEAN DEFAULT FALSE;
    """)
    
    conn.commit()
    print("✓ Successfully added ensemble_is_anomaly column to ml_results table")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
