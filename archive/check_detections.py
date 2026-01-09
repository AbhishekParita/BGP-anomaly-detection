"""
Quick script to check if anomaly detection is working
"""
import psycopg2
from datetime import datetime, timedelta

def check_detection_pipeline():
    """Check all detection tables for recent activity"""
    try:
        conn = psycopg2.connect(
            dbname="bgp_monitor",
            user="postgres",
            password="anand",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        
        print("="*70)
        print("DETECTION PIPELINE STATUS CHECK")
        print("="*70)
        
        # 1. Check raw data collection
        print("\n[1/5] Checking raw BGP data collection...")
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                MAX(timestamp) as latest,
                NOW() - MAX(timestamp) as age
            FROM raw_bgp_data
        """)
        total, latest, age = cursor.fetchone()
        print(f"✅ Total records: {total:,}")
        print(f"✅ Latest data: {latest}")
        print(f"⏰ Data age: {age}")
        
        # 2. Check feature aggregation
        print("\n[2/5] Checking feature aggregation...")
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                MAX(timestamp) as latest
            FROM features
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            print(f"✅ Feature windows: {result[0]:,}")
            print(f"✅ Latest window: {result[1]}")
        else:
            print("⚠️ No feature windows generated yet")
        
        # 3. Check ML detection results
        print("\n[3/5] Checking ML detection results...")
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                MAX(timestamp) as latest
            FROM ml_results
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            print(f"✅ ML detections: {result[0]:,}")
            print(f"✅ Latest detection: {result[1]}")
        else:
            print("⚠️ No ML detections yet")
        
        # 4. Check ensemble results
        print("\n[4/5] Checking ensemble coordination...")
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                MAX(timestamp) as latest
            FROM ensemble_results
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            print(f"✅ Ensemble results: {result[0]:,}")
            print(f"✅ Latest result: {result[1]}")
        else:
            print("⚠️ No ensemble results yet")
        
        # 5. Check final alerts
        print("\n[5/5] Checking final alerts...")
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                MAX(timestamp) as latest,
                COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical,
                COUNT(CASE WHEN severity = 'HIGH' THEN 1 END) as high,
                COUNT(CASE WHEN severity = 'MEDIUM' THEN 1 END) as medium
            FROM alerts
        """)
        result = cursor.fetchone()
        if result and result[0] > 0:
            total, latest, critical, high, medium = result
            print(f"✅ Total alerts: {total:,}")
            print(f"   - Critical: {critical}")
            print(f"   - High: {high}")
            print(f"   - Medium: {medium}")
            print(f"✅ Latest alert: {latest}")
        else:
            print("⚠️ No alerts generated yet")
        
        # Check recent alerts (last 10 minutes)
        print("\n" + "="*70)
        print("RECENT ALERTS (Last 10 minutes)")
        print("="*70)
        cursor.execute("""
            SELECT 
                timestamp,
                severity,
                alert_type,
                confidence
            FROM alerts
            WHERE timestamp > NOW() - INTERVAL '10 minutes'
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        recent = cursor.fetchall()
        if recent:
            for ts, sev, atype, conf in recent:
                print(f"{ts} | {sev:8s} | {atype:20s} | {conf:.2f}")
        else:
            print("No alerts in the last 10 minutes")
        
        print("\n" + "="*70)
        print("DIAGNOSIS:")
        print("="*70)
        
        # Diagnose the issue
        cursor.execute("SELECT COUNT(*) FROM features WHERE timestamp > NOW() - INTERVAL '5 minutes'")
        recent_features = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ml_results WHERE timestamp > NOW() - INTERVAL '5 minutes'")
        recent_ml = cursor.fetchone()[0]
        
        if total == 0:
            print("❌ PROBLEM: No raw data is being collected!")
            print("   → Check if RIS Live Collector is running")
        elif recent_features == 0:
            print("❌ PROBLEM: Feature Aggregator is not processing data!")
            print("   → Check if Feature Aggregator service is running")
        elif recent_ml == 0:
            print("❌ PROBLEM: ML detectors are not running!")
            print("   → Check if LSTM and Isolation Forest detectors are running")
        else:
            print("✅ All services appear to be working!")
            print("   → Dashboard should update within 30 seconds")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_detection_pipeline()
