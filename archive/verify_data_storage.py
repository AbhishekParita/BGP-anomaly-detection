"""
Verify Data is Actually Stored in Database
==========================================
"""

import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def verify_data_storage():
    """Check if data is actually in the database"""
    
    print("=" * 70)
    print("DATABASE STORAGE VERIFICATION")
    print("=" * 70)
    
    try:
        # Connect to database
        print("\n[1/4] Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'bgp_monitor'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'anand'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )
        cursor = conn.cursor()
        print("SUCCESS: Connected to database")
        
        # Count total records
        print("\n[2/4] Counting records in raw_bgp_data...")
        cursor.execute("SELECT COUNT(*) FROM raw_bgp_data;")
        total_count = cursor.fetchone()[0]
        print(f"SUCCESS: Found {total_count:,} total records")
        
        # Check for recent data (last 10 minutes)
        print("\n[3/4] Checking for recent data...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM raw_bgp_data 
            WHERE timestamp > NOW() - INTERVAL '10 minutes';
        """)
        recent_count = cursor.fetchone()[0]
        
        if recent_count > 0:
            print(f"SUCCESS: Found {recent_count} records from last 10 minutes")
            print("        ^^ This proves data is ACTIVELY being collected!")
        else:
            print("WARNING: No data from last 10 minutes")
            print("        (Collector might not be running)")
        
        # Show 5 most recent records
        print("\n[4/4] Showing 5 most recent records...")
        cursor.execute("""
            SELECT 
                timestamp,
                peer_addr,
                prefix,
                announcements,
                withdrawals,
                total_updates
            FROM raw_bgp_data 
            ORDER BY timestamp DESC 
            LIMIT 5;
        """)
        
        records = cursor.fetchall()
        
        if records:
            print("\n   TIME                 | PEER           | PREFIX          | ANN | WITH | TOTAL")
            print("   " + "-" * 80)
            for row in records:
                timestamp = row[0].strftime("%Y-%m-%d %H:%M:%S") if row[0] else "N/A"
                peer = row[1][:15] if row[1] else "N/A"
                prefix = row[2][:15] if row[2] else "N/A"
                ann = row[3] or 0
                wit = row[4] or 0
                tot = row[5] or 0
                print(f"   {timestamp} | {peer:15} | {prefix:15} | {ann:3} | {wit:4} | {tot:5}")
            
            print("\n" + "=" * 70)
            print("VERIFICATION COMPLETE!")
            print("=" * 70)
            print("\nCONCLUSION:")
            print(f"  ✅ Database connection: WORKING")
            print(f"  ✅ Total records: {total_count:,}")
            print(f"  ✅ Recent data: {recent_count} records")
            print(f"  ✅ Data storage: CONFIRMED")
            
            # Calculate data age
            cursor.execute("SELECT MAX(timestamp) FROM raw_bgp_data;")
            latest_time = cursor.fetchone()[0]
            if latest_time:
                age = datetime.now() - latest_time.replace(tzinfo=None)
                age_minutes = int(age.total_seconds() / 60)
                
                if age_minutes < 5:
                    print(f"  ✅ Data freshness: EXCELLENT (last update {age_minutes} minutes ago)")
                elif age_minutes < 30:
                    print(f"  ⚠️  Data freshness: OK (last update {age_minutes} minutes ago)")
                else:
                    print(f"  ⚠️  Data freshness: STALE (last update {age_minutes} minutes ago)")
                    print(f"     → Start the collector to get fresh data")
        else:
            print("   No records found in database")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_data_storage()
