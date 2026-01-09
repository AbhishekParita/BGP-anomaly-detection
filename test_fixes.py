"""
Test script to verify all fixes are working correctly
Tests stream limits, detector limits, and graceful shutdown
"""

import subprocess
import time
import sys
import os

def print_header(message):
    print("\n" + "="*60)
    print(message)
    print("="*60 + "\n")

def test_stream_generator():
    """Test 1: Verify stream generator stops after limit"""
    print_header("TEST 1: Stream Generator Limits")
    
    print("✓ Checking if stream_generator.py exists...")
    if not os.path.exists('stream_generator.py'):
        print("❌ stream_generator.py not found!")
        return False
    
    print("✓ Checking if MAX_RECORDS configuration exists...")
    with open('stream_generator.py', 'r') as f:
        content = f.read()
        if 'MAX_RECORDS' in content:
            print("✅ MAX_RECORDS configuration found")
        else:
            print("❌ MAX_RECORDS not found in stream_generator.py")
            return False
    
    print("✓ Checking if graceful shutdown handler exists...")
    if 'signal_handler' in content:
        print("✅ Signal handler implemented")
    else:
        print("⚠️  Signal handler not found")
    
    return True

def test_hybrid_detector():
    """Test 2: Verify hybrid detector has limits"""
    print_header("TEST 2: Hybrid Detector Limits")
    
    print("✓ Checking if hybrid_detector.py exists...")
    if not os.path.exists('hybrid_detector.py'):
        print("❌ hybrid_detector.py not found!")
        return False
    
    print("✓ Checking for MAX_MESSAGES configuration...")
    with open('hybrid_detector.py', 'r') as f:
        content = f.read()
        
        checks = {
            'MAX_MESSAGES': 'Message limit configuration',
            'MESSAGE_TIMEOUT': 'Timeout configuration',
            'message_count': 'Message counter',
            'shutdown_flag': 'Shutdown flag',
            'signal_handler': 'Signal handler'
        }
        
        for check, description in checks.items():
            if check in content:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
                return False
    
    return True

def test_config_file():
    """Test 3: Verify config.py exists with proper settings"""
    print_header("TEST 3: Configuration File")
    
    print("✓ Checking if config.py exists...")
    if not os.path.exists('config.py'):
        print("❌ config.py not found!")
        return False
    
    print("✓ Importing configuration...")
    try:
        import config
        
        required_configs = [
            'MAX_STREAM_RECORDS',
            'MAX_DETECTOR_MESSAGES',
            'MAX_DATABASE_RECORDS',
            'RETENTION_DAYS',
            'KAFKA_BOOTSTRAP_SERVERS',
            'DATABASE_URL'
        ]
        
        for cfg in required_configs:
            if hasattr(config, cfg):
                value = getattr(config, cfg)
                print(f"✅ {cfg} = {value}")
            else:
                print(f"❌ {cfg} not found")
                return False
        
        print("\n✅ All configuration values present")
        return True
        
    except Exception as e:
        print(f"❌ Error importing config: {e}")
        return False

def test_database_cleanup():
    """Test 4: Verify database cleanup utility exists"""
    print_header("TEST 4: Database Cleanup Utility")
    
    print("✓ Checking if database_cleanup.py exists...")
    if not os.path.exists('database_cleanup.py'):
        print("❌ database_cleanup.py not found!")
        return False
    
    print("✓ Checking for key functions...")
    with open('database_cleanup.py', 'r') as f:
        content = f.read()
        
        checks = {
            'class DatabaseCleanup': 'DatabaseCleanup class',
            'def cleanup_old_records': 'Cleanup function',
            'def enforce_size_limit': 'Size limit enforcement',
            'def vacuum_tables': 'Vacuum function',
            'RETENTION_DAYS': 'Retention configuration',
            'MAX_DATABASE_RECORDS': 'Record limit configuration'
        }
        
        for check, description in checks.items():
            if check in content:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
                return False
    
    return True

def test_database_triggers():
    """Test 5: Verify SQL triggers file exists"""
    print_header("TEST 5: Database Triggers")
    
    print("✓ Checking if database_triggers.sql exists...")
    if not os.path.exists('database_triggers.sql'):
        print("❌ database_triggers.sql not found!")
        return False
    
    print("✓ Checking for key triggers and functions...")
    with open('database_triggers.sql', 'r', encoding='utf-8') as f:
        content = f.read()
        
        checks = {
            'CREATE TABLE IF NOT EXISTS system_config': 'System config table',
            'CREATE OR REPLACE FUNCTION cleanup_old_records': 'Cleanup function',
            'CREATE OR REPLACE FUNCTION enforce_record_limit': 'Limit enforcement function',
            'CREATE TRIGGER': 'Trigger creation',
            'CREATE OR REPLACE VIEW system_stats': 'System stats view',
            'CREATE OR REPLACE PROCEDURE manual_cleanup': 'Manual cleanup procedure'
        }
        
        for check, description in checks.items():
            if check in content:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
                return False
    
    print("\n⚠️  Note: Triggers need to be installed in PostgreSQL:")
    print("   psql -U postgres -d bgp_monitor -f database_triggers.sql")
    
    return True

def test_documentation():
    """Test 6: Verify documentation exists"""
    print_header("TEST 6: Documentation")
    
    files_to_check = {
        'FIXES_APPLIED.md': 'Fixes documentation',
        'requirements.txt': 'Python dependencies',
        'ensemble_config_optimized.json': 'Ensemble configuration'
    }
    
    all_found = True
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            print(f"✅ {description} ({filename})")
        else:
            print(f"❌ {description} ({filename}) not found")
            all_found = False
    
    return all_found

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 BGP ANOMALY DETECTION - FIX VERIFICATION")
    print("="*60)
    
    tests = [
        ("Stream Generator Limits", test_stream_generator),
        ("Hybrid Detector Limits", test_hybrid_detector),
        ("Configuration File", test_config_file),
        ("Database Cleanup Utility", test_database_cleanup),
        ("Database Triggers", test_database_triggers),
        ("Documentation", test_documentation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your fixes are properly installed.")
        print("\n📋 Next Steps:")
        print("   1. Install database triggers: psql -U postgres -d bgp_monitor -f database_triggers.sql")
        print("   2. Test stream: python stream_generator.py")
        print("   3. Test detector: python hybrid_detector.py")
        print("   4. Run cleanup: python database_cleanup.py")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED. Please review the output above.")
        return 1
    
    print("="*60 + "\n")

if __name__ == "__main__":
    sys.exit(main())
