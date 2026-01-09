"""
Install Auto-Cleanup Triggers - Direct SQL (No File Reading)
=============================================================
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def install_triggers():
    """Install auto-cleanup triggers directly"""
    print("=" * 60)
    print("Installing Auto-Cleanup Triggers")
    print("=" * 60)
    
    try:
        # Connect
        print("\n[1/4] Connecting to PostgreSQL...")
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
        
        # Step 1: Create system_config table
        print("\n[2/4] Creating system_config table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                raw_bgp_data_limit INTEGER DEFAULT 50000,
                bgp_announcements_limit INTEGER DEFAULT 50000,
                ml_results_limit INTEGER DEFAULT 50000,
                ensemble_results_limit INTEGER DEFAULT 50000,
                retention_days INTEGER DEFAULT 30
            );
        """)
        
        cursor.execute("SELECT COUNT(*) FROM system_config;")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO system_config VALUES (50000, 50000, 50000, 50000, 30);
            """)
        
        conn.commit()
        print("SUCCESS: system_config table created")
        
        # Step 2: Create cleanup function
        print("\n[3/4] Creating cleanup functions...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION cleanup_old_records()
            RETURNS TRIGGER AS $func$
            DECLARE
                retention INTEGER;
            BEGIN
                SELECT retention_days INTO retention FROM system_config LIMIT 1;
                
                EXECUTE format(
                    'DELETE FROM %I WHERE timestamp < NOW() - INTERVAL ''%s days''',
                    TG_TABLE_NAME, retention
                );
                
                RETURN NULL;
            END;
            $func$ LANGUAGE plpgsql;
        """)
        
        cursor.execute("""
            CREATE OR REPLACE FUNCTION enforce_record_limit()
            RETURNS TRIGGER AS $func$
            DECLARE
                current_count BIGINT;
                limit_value INTEGER;
                excess_count INTEGER;
            BEGIN
                EXECUTE format('SELECT COUNT(*) FROM %I', TG_TABLE_NAME) INTO current_count;
                
                EXECUTE format(
                    'SELECT %I FROM system_config LIMIT 1',
                    TG_TABLE_NAME || '_limit'
                ) INTO limit_value;
                
                IF current_count > limit_value THEN
                    excess_count := current_count - limit_value;
                    
                    EXECUTE format(
                        'DELETE FROM %I WHERE id IN (
                            SELECT id FROM %I ORDER BY timestamp ASC LIMIT %s
                        )',
                        TG_TABLE_NAME, TG_TABLE_NAME, excess_count
                    );
                END IF;
                
                RETURN NULL;
            END;
            $func$ LANGUAGE plpgsql;
        """)
        
        conn.commit()
        print("SUCCESS: Cleanup functions created")
        
        # Step 3: Create triggers on each table
        print("\n[4/4] Creating triggers...")
        
        tables = ['raw_bgp_data', 'bgp_announcements', 'ml_results', 'ensemble_results']
        
        for table in tables:
            # Drop existing triggers first
            cursor.execute(f"DROP TRIGGER IF EXISTS trigger_cleanup_{table} ON {table};")
            cursor.execute(f"DROP TRIGGER IF EXISTS trigger_limit_{table} ON {table};")
            
            # Create cleanup trigger
            cursor.execute(f"""
                CREATE TRIGGER trigger_cleanup_{table}
                    AFTER INSERT ON {table}
                    FOR EACH STATEMENT
                    EXECUTE FUNCTION cleanup_old_records();
            """)
            
            # Create limit trigger
            cursor.execute(f"""
                CREATE TRIGGER trigger_limit_{table}
                    AFTER INSERT ON {table}
                    FOR EACH STATEMENT
                    EXECUTE FUNCTION enforce_record_limit();
            """)
            
            print(f"   OK: Triggers created for {table}")
        
        conn.commit()
        print("SUCCESS: All triggers created")
        
        # Verify
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        
        cursor.execute("SELECT * FROM system_config;")
        config = cursor.fetchone()
        if config:
            print("   OK: system_config table")
            print(f"      - Raw BGP limit: {config[0]:,} records")
            print(f"      - Announcements limit: {config[1]:,} records")
            print(f"      - ML results limit: {config[2]:,} records")
            print(f"      - Ensemble limit: {config[3]:,} records")
            print(f"      - Retention: {config[4]} days")
        
        # Count triggers
        cursor.execute("""
            SELECT COUNT(*) FROM pg_trigger 
            WHERE tgname LIKE 'trigger_cleanup_%' OR tgname LIKE 'trigger_limit_%';
        """)
        trigger_count = cursor.fetchone()[0]
        print(f"   OK: {trigger_count} triggers installed")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("SUCCESS: Auto-cleanup triggers installed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    import sys
    success = install_triggers()
    sys.exit(0 if success else 1)
