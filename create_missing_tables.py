"""
Create Missing Database Tables - Simple Version (No Emojis)
============================================================
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def create_tables():
    """Create missing tables"""
    print("=" * 60)
    print("Creating Missing Database Tables")
    print("=" * 60)
    
    try:
        # Connect
        print("\n[1/3] Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'bgp_monitor'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'anand'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )
        conn.autocommit = False
        cursor = conn.cursor()
        print("SUCCESS: Connected")
        
        # Read schema file
        print("\n[2/3] Reading schema file...")
        with open('database/schema_complete.sql', 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        print("SUCCESS: Schema loaded")
        
        # Execute (ignore errors for existing objects)
        print("\n[3/3] Creating tables...")
        print("   This may take 30-60 seconds...")
        print("   (Ignoring errors for existing objects)")
        
        # Split SQL by statement and execute one by one
        statements = schema_sql.split(';')
        created = 0
        skipped = 0
        
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            
            try:
                cursor.execute(statement)
                conn.commit()
                created += 1
            except psycopg2.Error as e:
                # Ignore "already exists" errors
                if 'already exists' in str(e):
                    skipped += 1
                    conn.rollback()
                else:
                    print(f"   WARNING: {e}")
                    conn.rollback()
        
        print(f"SUCCESS: Created {created} objects, skipped {skipped} existing")
        
        # Verify
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        
        tables = ['raw_bgp_data', 'bgp_announcements', 'ml_results', 
                  'ensemble_results', 'alerts', 'detections']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"   OK: {table} ({count:,} records)")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("SUCCESS: All tables created!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    import sys
    success = create_tables()
    sys.exit(0 if success else 1)
