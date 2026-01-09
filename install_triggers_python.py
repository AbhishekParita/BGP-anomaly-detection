"""
Install Database Triggers using Python (No psql required)
===========================================================
Installs auto-cleanup triggers from database_triggers.sql
"""

import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def install_triggers():
    """Install triggers from SQL file using Python"""
    
    # Connect to database
    try:
        print("🔌 Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'bgp_monitor'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'anand'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )
        conn.autocommit = False
        cursor = conn.cursor()
        print("✅ Connected!")
        
        # Read trigger SQL file
        trigger_file = Path('database_triggers.sql')
        if not trigger_file.exists():
            print(f"❌ Trigger file not found: {trigger_file}")
            return False
        
        print(f"\n📄 Reading trigger file: {trigger_file}")
        with open(trigger_file, 'r', encoding='utf-8') as f:
            trigger_sql = f.read()
        
        print("⚙️  Installing triggers...")
        print("   - system_config table")
        print("   - cleanup_old_records() function")
        print("   - enforce_record_limit() function")
        print("   - Triggers on all tables")
        print("")
        
        # Execute SQL
        cursor.execute(trigger_sql)
        conn.commit()
        
        print("✅ Triggers installed successfully!")
        
        # Verify
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'system_config'
            );
        """)
        has_config = cursor.fetchone()[0]
        
        if has_config:
            print("\n✓ Verification:")
            print("   ✅ system_config table created")
            
            # Show limits
            cursor.execute("SELECT * FROM system_config;")
            config = cursor.fetchone()
            if config:
                print(f"   ✅ Limits configured:")
                print(f"      - Raw BGP data: {config[0]:,} records")
                print(f"      - BGP announcements: {config[1]:,} records")
                print(f"      - ML results: {config[2]:,} records")
                print(f"      - Ensemble results: {config[3]:,} records")
                print(f"      - Retention: {config[4]} days")
        else:
            print("⚠️  system_config table not found after installation")
        
        cursor.close()
        conn.close()
        print("\n✅ Trigger installation complete!")
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Error installing triggers: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = install_triggers()
    sys.exit(0 if success else 1)
