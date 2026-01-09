"""
Install Database Triggers - Simple Version (No Emojis)
=======================================================
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def install_triggers():
    """Install auto-cleanup triggers"""
    print("=" * 60)
    print("Installing Auto-Cleanup Triggers")
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
        
        # Read trigger file
        print("\n[2/3] Reading trigger file...")
        with open('database_triggers.sql', 'r', encoding='utf-8') as f:
            trigger_sql = f.read()
        print("SUCCESS: Trigger file loaded")
        
        # Execute
        print("\n[3/3] Installing triggers...")
        cursor.execute(trigger_sql)
        conn.commit()
        print("SUCCESS: Triggers installed")
        
        # Verify
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        
        cursor.execute("SELECT * FROM system_config;")
        config = cursor.fetchone()
        if config:
            print("   OK: system_config table created")
            print(f"   OK: Raw BGP limit: {config[0]:,} records")
            print(f"   OK: Announcements limit: {config[1]:,} records")
            print(f"   OK: Retention: {config[4]} days")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("SUCCESS: Triggers installed!")
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
    success = install_triggers()
    sys.exit(0 if success else 1)
