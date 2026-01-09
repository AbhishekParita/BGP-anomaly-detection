"""
Database Initialization Script
================================
Purpose: Initialize PostgreSQL database with complete schema for BGP Anomaly Detection

Usage:
    python database/init_database.py

What this script does:
1. Connects to PostgreSQL database
2. Creates all required tables with TimescaleDB extensions
3. Sets up indexes, views, and functions
4. Applies retention policies
5. Validates the setup

Requirements:
- PostgreSQL 14+ running
- TimescaleDB extension installed
- Database credentials in .env file
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class DatabaseInitializer:
    """Handles database initialization and schema creation"""
    
    def __init__(self):
        """Initialize with database credentials from environment"""
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        self.conn = None
        self.cursor = None
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            print("🔍 Checking if database exists...")
            
            # Connect to default 'postgres' database to create our database
            default_config = self.db_config.copy()
            default_config['dbname'] = 'postgres'
            
            conn = psycopg2.connect(**default_config)
            conn.autocommit = True  # Required for CREATE DATABASE
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_database WHERE datname = %s
                );
            """, (self.db_config['dbname'],))
            
            exists = cursor.fetchone()[0]
            
            if exists:
                print(f"✅ Database '{self.db_config['dbname']}' already exists")
            else:
                print(f"📝 Creating database '{self.db_config['dbname']}'...")
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.db_config['dbname'])
                ))
                print(f"✅ Database '{self.db_config['dbname']}' created successfully!")
            
            cursor.close()
            conn.close()
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Database creation failed: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check if PostgreSQL is running")
            print("   2. Verify credentials in .env file")
            print("   3. Ensure user has CREATE DATABASE privilege")
            return False
        
    def connect(self):
        """Establish database connection"""
        try:
            print(f"\n🔌 Connecting to database '{self.db_config['dbname']}'...")
            print(f"   Host: {self.db_config['host']}:{self.db_config['port']}")
            
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            
            print("✅ Connected successfully!")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def check_timescaledb(self):
        """Check if TimescaleDB extension is available"""
        try:
            print("\n🔍 Checking TimescaleDB extension...")
            
            # Check if extension exists
            self.cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_available_extensions 
                    WHERE name = 'timescaledb'
                );
            """)
            
            available = self.cursor.fetchone()[0]
            
            if not available:
                print("⚠️  TimescaleDB extension not found!")
                print("💡 Install TimescaleDB: https://docs.timescale.com/install/")
                return False
            
            # Check if extension is enabled
            self.cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension 
                    WHERE extname = 'timescaledb'
                );
            """)
            
            enabled = self.cursor.fetchone()[0]
            
            if enabled:
                print("✅ TimescaleDB extension is enabled")
            else:
                print("⚠️  TimescaleDB is available but not enabled")
                print("   Attempting to enable...")
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                self.conn.commit()
                print("✅ TimescaleDB enabled successfully!")
            
            return True
            
        except psycopg2.Error as e:
            print(f"❌ TimescaleDB check failed: {e}")
            return False
    
    def execute_schema_file(self, schema_path):
        """Execute SQL schema file"""
        try:
            print(f"\n📄 Loading schema from: {schema_path}")
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            print("⚙️  Executing schema creation...")
            print("   This may take a minute...")
            
            # Execute the entire schema
            self.cursor.execute(schema_sql)
            self.conn.commit()
            
            print("✅ Schema executed successfully!")
            return True
            
        except FileNotFoundError:
            print(f"❌ Schema file not found: {schema_path}")
            return False
        except psycopg2.Error as e:
            print(f"❌ Schema execution failed: {e}")
            self.conn.rollback()
            return False
    
    def verify_tables(self):
        """Verify that all required tables were created"""
        required_tables = [
            'raw_bgp_data',
            'features',
            'ml_results',
            'route_monitor_events',
            'alerts',
            'system_metrics'
        ]
        
        print("\n✓ Verifying table creation...")
        
        try:
            for table in required_tables:
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                
                exists = self.cursor.fetchone()[0]
                
                if exists:
                    # Get row count
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    count = self.cursor.fetchone()[0]
                    print(f"   ✅ {table}: Created (rows: {count})")
                else:
                    print(f"   ❌ {table}: Missing!")
                    return False
            
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def verify_views(self):
        """Verify that views were created"""
        required_views = ['recent_alerts', 'alert_summary_hourly']
        
        print("\n✓ Verifying views...")
        
        try:
            for view in required_views:
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.views 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (view,))
                
                exists = self.cursor.fetchone()[0]
                status = "✅" if exists else "❌"
                print(f"   {status} {view}")
            
            return True
            
        except psycopg2.Error as e:
            print(f"❌ View verification failed: {e}")
            return False
    
    def verify_functions(self):
        """Verify that functions were created"""
        required_functions = [
            'calculate_ensemble_score',
            'update_alert_status'
        ]
        
        print("\n✓ Verifying functions...")
        
        try:
            for func in required_functions:
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_proc 
                        WHERE proname = %s
                    );
                """, (func,))
                
                exists = self.cursor.fetchone()[0]
                status = "✅" if exists else "❌"
                print(f"   {status} {func}")
            
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Function verification failed: {e}")
            return False
    
    def get_database_size(self):
        """Get total database size"""
        try:
            self.cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(%s)) as size;
            """, (self.db_config['dbname'],))
            
            size = self.cursor.fetchone()[0]
            print(f"\n📊 Database size: {size}")
            
        except psycopg2.Error as e:
            print(f"⚠️  Could not get database size: {e}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("\n🔌 Database connection closed")
    
    def initialize(self):
        """Main initialization workflow"""
        print("=" * 60)
        print("BGP ANOMALY DETECTION - DATABASE INITIALIZATION")
        print("=" * 60)
        
        # Step 1: Create database if it doesn't exist
        if not self.create_database_if_not_exists():
            return False
        
        # Step 2: Connect to the database
        if not self.connect():
            return False
        
        # Step 3: Check TimescaleDB
        if not self.check_timescaledb():
            return False
        
        # Step 4: Execute schema
        schema_path = Path(__file__).parent / 'schema_complete.sql'
        if not self.execute_schema_file(schema_path):
            return False
        
        # Step 5: Verify setup
        tables_ok = self.verify_tables()
        views_ok = self.verify_views()
        functions_ok = self.verify_functions()
        
        if tables_ok and views_ok and functions_ok:
            print("\n" + "=" * 60)
            print("✅ DATABASE INITIALIZATION COMPLETE!")
            print("=" * 60)
            
            self.get_database_size()
            
            print("\n📋 Next Steps:")
            print("   1. Run sample data script: python database/insert_sample_data.py")
            print("   2. Start RIS Live collector: python services/ris_live_collector.py")
            print("   3. Start detection service: python services/detection_service.py")
            print("   4. Start API server: python api/main.py")
            print("\n" + "=" * 60)
            return True
        else:
            print("\n❌ Initialization completed with errors")
            return False


def main():
    """Main entry point"""
    initializer = DatabaseInitializer()
    
    try:
        success = initializer.initialize()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        initializer.close()


if __name__ == "__main__":
    main()
