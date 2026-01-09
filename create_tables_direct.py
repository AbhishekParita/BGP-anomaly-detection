"""
Create Only Missing Tables - Direct SQL
========================================
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def create_missing_tables():
    """Create only the missing tables"""
    print("=" * 60)
    print("Creating Missing Database Tables")
    print("=" * 60)
    
    try:
        # Connect
        print("\n[1/2] Connecting to PostgreSQL...")
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
        
        # Check which tables are missing
        print("\n[2/2] Checking for missing tables...")
        
        tables_to_create = {
            'bgp_announcements': """
                CREATE TABLE IF NOT EXISTS bgp_announcements (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    peer_addr TEXT NOT NULL,
                    peer_asn BIGINT,
                    prefix CIDR,
                    as_path TEXT,
                    origin_asn BIGINT,
                    next_hop TEXT,
                    communities TEXT[],
                    is_withdrawal BOOLEAN DEFAULT FALSE,
                    raw_message JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """,
            'features': """
                CREATE TABLE IF NOT EXISTS features (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    peer_addr TEXT NOT NULL,
                    prefix CIDR,
                    announcements INTEGER DEFAULT 0,
                    withdrawals INTEGER DEFAULT 0,
                    total_updates INTEGER DEFAULT 0,
                    withdrawal_ratio REAL DEFAULT 0.0,
                    flap_count INTEGER DEFAULT 0,
                    path_length REAL DEFAULT 0.0,
                    unique_peers INTEGER DEFAULT 0,
                    message_rate REAL DEFAULT 0.0,
                    session_resets INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """,
            'ml_results': """
                CREATE TABLE IF NOT EXISTS ml_results (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    peer_addr TEXT NOT NULL,
                    prefix CIDR,
                    model_type TEXT NOT NULL,
                    anomaly_score REAL,
                    is_anomaly BOOLEAN,
                    confidence REAL,
                    feature_importances JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """,
            'ensemble_results': """
                CREATE TABLE IF NOT EXISTS ensemble_results (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    peer_addr TEXT NOT NULL,
                    prefix CIDR,
                    lstm_score REAL,
                    if_score REAL,
                    heuristic_score REAL,
                    ensemble_score REAL,
                    final_decision BOOLEAN,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """,
            'alerts': """
                CREATE TABLE IF NOT EXISTS alerts (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    peer_addr TEXT NOT NULL,
                    prefix CIDR,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    ensemble_score REAL,
                    confidence REAL,
                    status TEXT DEFAULT 'new',
                    acknowledged_by TEXT,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """,
            'detections': """
                CREATE TABLE IF NOT EXISTS detections (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    peer_addr TEXT NOT NULL,
                    prefix CIDR,
                    detection_type TEXT NOT NULL,
                    anomaly_score REAL,
                    confidence REAL,
                    details JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """
        }
        
        created_count = 0
        for table_name, create_sql in tables_to_create.items():
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table_name,))
            
            exists = cursor.fetchone()[0]
            
            if not exists:
                print(f"   Creating: {table_name}...")
                cursor.execute(create_sql)
                conn.commit()
                created_count += 1
                print(f"   SUCCESS: {table_name} created")
            else:
                print(f"   SKIP: {table_name} already exists")
        
        # Verify all tables
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        
        all_tables = ['raw_bgp_data', 'bgp_announcements', 'features',
                      'ml_results', 'ensemble_results', 'alerts', 'detections']
        
        for table in all_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                print(f"   OK: {table} ({count:,} records)")
            except Exception as e:
                print(f"   ERROR: {table} - {e}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print(f"SUCCESS: Created {created_count} new tables!")
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
    success = create_missing_tables()
    sys.exit(0 if success else 1)
