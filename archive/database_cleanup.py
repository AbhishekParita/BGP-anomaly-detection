"""
Database Cleanup Utility
Automatically cleans old records and maintains database size limits
Prevents infinite data accumulation and disk space exhaustion
"""

import psycopg2
from datetime import datetime, timedelta
import time
import sys
import os
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Import config
try:
    from config import (
        MAX_DATABASE_RECORDS, 
        RETENTION_DAYS, 
        CLEANUP_BATCH_SIZE
    )
except ImportError:
    # Fallback to default values
    MAX_DATABASE_RECORDS = 100000
    RETENTION_DAYS = 30
    CLEANUP_BATCH_SIZE = 1000

class DatabaseCleanup:
    """Manages database cleanup operations"""
    
    def __init__(self):
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.conn = None
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            print("✅ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def get_table_count(self, table_name):
        """Get total record count for a table"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                return count
        except Exception as e:
            print(f"⚠️ Error counting records in {table_name}: {e}")
            return 0
    
    def cleanup_old_records(self, table_name, date_column='timestamp'):
        """Delete records older than RETENTION_DAYS"""
        try:
            cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
            
            with self.conn.cursor() as cur:
                # Count records to delete
                cur.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {date_column} < %s",
                    (cutoff_date,)
                )
                delete_count = cur.fetchone()[0]
                
                if delete_count == 0:
                    print(f"   ✓ {table_name}: No old records to delete")
                    return 0
                
                # Delete in batches
                total_deleted = 0
                while True:
                    cur.execute(
                        f"""DELETE FROM {table_name} 
                            WHERE {date_column} < %s 
                            LIMIT %s""",
                        (cutoff_date, CLEANUP_BATCH_SIZE)
                    )
                    deleted = cur.rowcount
                    total_deleted += deleted
                    self.conn.commit()
                    
                    if deleted < CLEANUP_BATCH_SIZE:
                        break
                    
                    print(f"   ... deleted {total_deleted}/{delete_count} records")
                
                print(f"   ✓ {table_name}: Deleted {total_deleted} old records (>{RETENTION_DAYS} days)")
                return total_deleted
                
        except Exception as e:
            print(f"   ❌ Error cleaning {table_name}: {e}")
            self.conn.rollback()
            return 0
    
    def enforce_size_limit(self, table_name, date_column='timestamp'):
        """Enforce maximum record count limit"""
        try:
            current_count = self.get_table_count(table_name)
            
            if current_count <= MAX_DATABASE_RECORDS:
                print(f"   ✓ {table_name}: Within limit ({current_count:,}/{MAX_DATABASE_RECORDS:,})")
                return 0
            
            # Delete oldest records to get under limit
            excess = current_count - MAX_DATABASE_RECORDS
            
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""DELETE FROM {table_name} 
                        WHERE {date_column} IN (
                            SELECT {date_column} FROM {table_name} 
                            ORDER BY {date_column} ASC 
                            LIMIT %s
                        )""",
                    (excess,)
                )
                deleted = cur.rowcount
                self.conn.commit()
                
                print(f"   ✓ {table_name}: Deleted {deleted:,} oldest records (limit enforcement)")
                return deleted
                
        except Exception as e:
            print(f"   ❌ Error enforcing limit on {table_name}: {e}")
            self.conn.rollback()
            return 0
    
    def vacuum_tables(self, table_names):
        """Run VACUUM to reclaim disk space"""
        try:
            # VACUUM needs autocommit mode
            old_isolation = self.conn.isolation_level
            self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            
            with self.conn.cursor() as cur:
                for table in table_names:
                    print(f"   🧹 Vacuuming {table}...")
                    cur.execute(f"VACUUM ANALYZE {table}")
            
            self.conn.set_isolation_level(old_isolation)
            print("   ✅ Vacuum completed")
            
        except Exception as e:
            print(f"   ⚠️ Vacuum error: {e}")
    
    def run_full_cleanup(self):
        """Run complete cleanup process"""
        print("\n" + "="*60)
        print("🧹 BGP ANOMALY DETECTION - DATABASE CLEANUP")
        print("="*60)
        print(f"⚙️  Configuration:")
        print(f"   - Max records per table: {MAX_DATABASE_RECORDS:,}")
        print(f"   - Retention period: {RETENTION_DAYS} days")
        print(f"   - Batch size: {CLEANUP_BATCH_SIZE:,}")
        print()
        
        if not self.connect():
            return False
        
        # Define tables to clean
        tables = [
            ('raw_bgp_data', 'timestamp'),
            ('bgp_announcements', 'timestamp'),
            ('ml_results', 'timestamp'),
            ('ensemble_results', 'timestamp')
        ]
        
        total_deleted = 0
        
        for table_name, date_column in tables:
            print(f"\n📊 Processing: {table_name}")
            
            # Step 1: Show current count
            count = self.get_table_count(table_name)
            print(f"   Current records: {count:,}")
            
            # Step 2: Delete old records
            deleted = self.cleanup_old_records(table_name, date_column)
            total_deleted += deleted
            
            # Step 3: Enforce size limit
            deleted = self.enforce_size_limit(table_name, date_column)
            total_deleted += deleted
            
            # Step 4: Show final count
            new_count = self.get_table_count(table_name)
            print(f"   Final records: {new_count:,}")
        
        # Vacuum to reclaim space
        print("\n🧹 Reclaiming disk space...")
        self.vacuum_tables([t[0] for t in tables])
        
        print("\n" + "="*60)
        print(f"✅ CLEANUP COMPLETE")
        print(f"   Total records deleted: {total_deleted:,}")
        print("="*60 + "\n")
        
        return True
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")

def main():
    """Main execution"""
    cleanup = DatabaseCleanup()
    
    try:
        success = cleanup.run_full_cleanup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup.close()

if __name__ == "__main__":
    main()
