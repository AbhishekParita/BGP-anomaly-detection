"""
Complete System Diagnosis for BGP Anomaly Detection
Checks all components according to RIS Live architecture
"""

import os
import sys
import psycopg2
from datetime import datetime, timedelta
import json
from pathlib import Path

def print_header(title, char="="):
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}\n")

def print_status(message, status="info"):
    symbols = {
        "pass": "✅",
        "fail": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }
    print(f"{symbols.get(status, 'ℹ️')} {message}")

# =============================================================================
# DATABASE CHECKS
# =============================================================================

def check_database_connection():
    """Check PostgreSQL connection"""
    print_header("1. DATABASE CONNECTION", "=")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "bgp_monitor"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        print_status("PostgreSQL connection successful", "pass")
        return conn
    except Exception as e:
        print_status(f"Database connection failed: {e}", "fail")
        return None

def check_database_tables(conn):
    """Check if required tables exist"""
    print_header("2. DATABASE TABLES", "=")
    
    required_tables = [
        'raw_bgp_data',
        'bgp_announcements',
        'ml_results',
        'ensemble_results',
        'alerts',
        'detections'
    ]
    
    cursor = conn.cursor()
    all_exist = True
    
    for table in required_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print_status(f"Table '{table}' exists with {count:,} records", "pass")
        except Exception as e:
            print_status(f"Table '{table}' missing or error: {e}", "fail")
            all_exist = False
    
    cursor.close()
    return all_exist

def check_database_triggers(conn):
    """Check if auto-cleanup triggers are installed"""
    print_header("3. DATABASE TRIGGERS (Auto-Cleanup)", "=")
    
    cursor = conn.cursor()
    
    # Check for system_config table
    try:
        cursor.execute("SELECT COUNT(*) FROM system_config")
        count = cursor.fetchone()[0]
        print_status(f"system_config table exists with {count} configs", "pass")
        
        # Show config values (our table has specific column names)
        cursor.execute("""
            SELECT raw_bgp_data_limit, bgp_announcements_limit, 
                   ml_results_limit, ensemble_results_limit, retention_days 
            FROM system_config LIMIT 1
        """)
        config = cursor.fetchone()
        if config:
            print("\n   Configuration Limits:")
            print(f"   - Raw BGP data: {config[0]:,} records")
            print(f"   - BGP announcements: {config[1]:,} records")
            print(f"   - ML results: {config[2]:,} records")
            print(f"   - Ensemble results: {config[3]:,} records")
            print(f"   - Retention period: {config[4]} days")
        
    except Exception as e:
        print_status(f"system_config table not found: {e}", "fail")
        print_status("Run: python install_triggers_direct.py", "warning")
        cursor.close()
        return False
    
    # Check for triggers
    try:
        cursor.execute("""
            SELECT trigger_name, event_object_table 
            FROM information_schema.triggers 
            WHERE trigger_schema = 'public'
            ORDER BY event_object_table, trigger_name
        """)
        triggers = cursor.fetchall()
        
        if triggers:
            print(f"\n   Installed Triggers ({len(triggers)}):")
            for trigger_name, table_name in triggers:
                print(f"   - {trigger_name} on {table_name}")
            print_status(f"Found {len(triggers)} auto-cleanup triggers", "pass")
        else:
            print_status("No triggers found - auto-cleanup NOT active", "warning")
            return False
            
    except Exception as e:
        print_status(f"Error checking triggers: {e}", "fail")
        cursor.close()
        return False
    
    cursor.close()
    return True

def check_recent_data(conn):
    """Check if data is actively coming in"""
    print_header("4. DATA FLOW (Recent Activity)", "=")
    
    cursor = conn.cursor()
    
    # Check each table for recent data
    tables_to_check = [
        ('raw_bgp_data', 'timestamp'),
        ('bgp_announcements', 'timestamp'),
        ('alerts', 'timestamp'),  # Fixed: alerts uses 'timestamp' not 'detected_at'
        ('detections', 'timestamp')
    ]
    
    for table, time_col in tables_to_check:
        try:
            # Get most recent record
            cursor.execute(f"""
                SELECT {time_col}, COUNT(*) as total
                FROM {table}
                GROUP BY {time_col}
                ORDER BY {time_col} DESC
                LIMIT 1
            """)
            result = cursor.fetchone()
            
            if result:
                last_time, count = result
                age = datetime.now() - last_time
                
                if age.total_seconds() < 300:  # Less than 5 minutes
                    print_status(f"{table}: Latest data {age.seconds}s ago", "pass")
                elif age.total_seconds() < 3600:  # Less than 1 hour
                    print_status(f"{table}: Latest data {age.seconds//60}m ago", "warning")
                else:
                    print_status(f"{table}: Latest data {age} ago (STALE)", "fail")
            else:
                print_status(f"{table}: No data found", "warning")
                
        except Exception as e:
            print_status(f"{table}: Error - {e}", "fail")
    
    cursor.close()

# =============================================================================
# ARCHITECTURE CHECKS
# =============================================================================

def check_ris_live_architecture():
    """Check if RIS Live architecture files exist"""
    print_header("5. RIS LIVE ARCHITECTURE", "=")
    
    architecture_files = {
        'routinator/ris_live_client (1).py': 'RIS Live WebSocket client',
        'routinator/routinator_client.py': 'RPKI/Routinator client',
        'routinator/database.py': 'Database ORM',
        'routinator/main (1).py': 'Test data generator',
        'services/ris_live_collector.py': 'RIS Live collector service',
        'services/feature_aggregator.py': 'Feature aggregation service',
        'services/heuristic_detector.py': 'Heuristic detection service',
        'services/lstm_detector.py': 'LSTM detection service',
        'services/isolation_forest_detector.py': 'Isolation Forest service',
        'services/correlation_engine.py': 'Correlation engine',
        'services/ensemble_coordinator.py': 'Ensemble coordinator',
        'run_all_services.py': 'Service orchestrator',
        'api.py': 'REST API server',
        'dashboard/index.html': 'Web dashboard'
    }
    
    all_exist = True
    for file_path, description in architecture_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print_status(f"{description}: {size:,} bytes", "pass")
        else:
            print_status(f"{description}: MISSING", "fail")
            all_exist = False
    
    return all_exist

def check_ml_models():
    """Check if ML models are present"""
    print_header("6. ML MODELS", "=")
    
    model_files = {
        'model_output/lstm/lstm_best.h5': 'LSTM Autoencoder model',
        'model_artifacts/iso_forest_bgp_production.pkl': 'Isolation Forest model',  # Fixed path
        'model_output/scaler.pkl': 'Feature scaler',
        'ensemble_config_optimized.json': 'Ensemble configuration'
    }
    
    all_exist = True
    for file_path, description in model_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print_status(f"{description}: {size:,} bytes", "pass")
        else:
            print_status(f"{description}: MISSING", "warning")
            all_exist = False
    
    return all_exist

def check_configuration_files():
    """Check configuration files"""
    print_header("7. CONFIGURATION FILES", "=")
    
    config_files = {
        'config.py': 'System limits configuration',
        'config/system_limits.json': 'JSON limits config',
        '.env': 'Environment variables',
        'requirements.txt': 'Python dependencies'
    }
    
    for file_path, description in config_files.items():
        if os.path.exists(file_path):
            print_status(f"{description}: Present", "pass")
        else:
            print_status(f"{description}: MISSING", "warning")

def check_limits_in_code():
    """Check if limits are implemented in code"""
    print_header("8. LIMIT IMPLEMENTATIONS", "=")
    
    # Check RIS Live client
    ris_client_path = 'routinator/ris_live_client (1).py'
    if os.path.exists(ris_client_path):
        with open(ris_client_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'max_messages': 'Message limit',
            'max_runtime': 'Runtime limit',
            'message_count': 'Message counter',
            'self.running = False': 'Stop mechanism'
        }
        
        print("   RIS Live Client:")
        for check, desc in checks.items():
            if check in content:
                print_status(f"  {desc} implemented", "pass")
            else:
                print_status(f"  {desc} MISSING", "fail")
    
    # Check test data generator
    test_gen_path = 'routinator/main (1).py'
    if os.path.exists(test_gen_path):
        with open(test_gen_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'max_records': 'Record limit',
            'stop_flag': 'Stop flag'
        }
        
        print("\n   Test Data Generator:")
        for check, desc in checks.items():
            if check in content:
                print_status(f"  {desc} implemented", "pass")
            else:
                print_status(f"  {desc} MISSING", "fail")

def check_obsolete_kafka_files():
    """Check for obsolete Kafka files (should be removed)"""
    print_header("9. OBSOLETE FILES CHECK", "=")
    
    obsolete_files = [
        'kafka_2.13-3.9.0/',
        'stream_generator.py',
        'hybrid_detector.py',
        'bmp_generator.py'
    ]
    
    found_obsolete = []
    for file_path in obsolete_files:
        if os.path.exists(file_path):
            found_obsolete.append(file_path)
            print_status(f"Found obsolete file: {file_path}", "warning")
    
    if not found_obsolete:
        print_status("No obsolete Kafka files found (Good!)", "pass")
    else:
        print_status(f"Found {len(found_obsolete)} obsolete files - consider removing", "warning")

def check_resource_usage(conn):
    """Check resource usage vs limits"""
    print_header("10. RESOURCE USAGE", "=")
    
    cursor = conn.cursor()
    
    # Check database sizes
    try:
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                pg_total_relation_size(schemaname||'.'||tablename) AS bytes
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY bytes DESC
            LIMIT 10
        """)
        
        print("   Database Table Sizes:")
        for schema, table, size, bytes_val in cursor.fetchall():
            print(f"   - {table}: {size}")
        
    except Exception as e:
        print_status(f"Error checking sizes: {e}", "fail")
    
    # Check record counts vs limits
    try:
        cursor.execute("""
            SELECT raw_bgp_data_limit, bgp_announcements_limit 
            FROM system_config LIMIT 1
        """)
        config = cursor.fetchone()
        
        if config:
            tables_to_check = {
                'raw_bgp_data': config[0],
                'bgp_announcements': config[1]
            }
            
            print("\n   Record Counts vs Limits:")
            for table, limit in tables_to_check.items():
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                usage_pct = (count / limit) * 100
                
                # More realistic thresholds: warn at 95%, fail only if OVER limit
                if count > limit:
                    status = "fail"
                elif usage_pct >= 95:
                    status = "warning"
                else:
                    status = "pass"
                    
                print_status(f"  {table}: {count:,} / {limit:,} ({usage_pct:.1f}%)", status)
    
    except Exception as e:
        print_status(f"Error checking limits: {e}", "warning")
    
    cursor.close()

# =============================================================================
# MAIN DIAGNOSIS
# =============================================================================

def run_complete_diagnosis():
    """Run complete system diagnosis"""
    print("\n" + "="*70)
    print("  BGP ANOMALY DETECTION - COMPLETE SYSTEM DIAGNOSIS")
    print("  Based on RIS Live Architecture")
    print("="*70)
    print(f"\n  Diagnosis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Database checks
    conn = check_database_connection()
    results['database_connection'] = conn is not None
    
    if conn:
        results['database_tables'] = check_database_tables(conn)
        results['database_triggers'] = check_database_triggers(conn)
        check_recent_data(conn)
        check_resource_usage(conn)
    
    # Architecture checks
    results['ris_live_architecture'] = check_ris_live_architecture()
    results['ml_models'] = check_ml_models()
    check_configuration_files()
    check_limits_in_code()
    check_obsolete_kafka_files()
    
    # Final summary
    print_header("DIAGNOSIS SUMMARY", "=")
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    print(f"\n   Results: {passed_checks}/{total_checks} critical checks passed")
    print("\n   Component Status:")
    
    status_map = {
        'database_connection': 'Database Connection',
        'database_tables': 'Database Tables',
        'database_triggers': 'Auto-Cleanup Triggers',
        'ris_live_architecture': 'RIS Live Architecture',
        'ml_models': 'ML Models'
    }
    
    for key, description in status_map.items():
        if key in results:
            status = "pass" if results[key] else "fail"
            print_status(f"  {description}", status)
    
    print("\n" + "="*70)
    
    if passed_checks == total_checks:
        print("\n  🎉 ALL SYSTEMS OPERATIONAL!")
        print("  ✅ Your BGP Anomaly Detection system is fully configured")
        print("  ✅ Following RIS Live architecture correctly")
        print("  ✅ Auto-cleanup triggers installed")
        print("  ✅ Ready for production use")
    else:
        print("\n  ⚠️  SOME ISSUES DETECTED")
        print(f"  {passed_checks}/{total_checks} checks passed")
        print("  Review the output above for details")
    
    print("\n" + "="*70 + "\n")
    
    if conn:
        conn.close()

if __name__ == "__main__":
    try:
        run_complete_diagnosis()
    except KeyboardInterrupt:
        print("\n\n🛑 Diagnosis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
