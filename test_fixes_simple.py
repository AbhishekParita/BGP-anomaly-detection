"""
Simple test to verify fixes without needing API server
Tests configuration, file existence, and database cleanup utility
"""

import os
import sys
from pathlib import Path

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_files_exist():
    """Test that all fixed files exist"""
    print_header("TEST 1: Fixed Files Exist")
    
    required_files = {
        'config.py': 'Configuration file with limits',
        'database_cleanup.py': 'Database cleanup utility',
        'database_triggers.sql': 'SQL triggers for auto-cleanup',
        'routinator/ris_live_client (1).py': 'RIS Live client with limits',
        'routinator/main (1).py': 'Test data generator with limits',
        'README_COMPLETE.md': 'Complete documentation',
        'test_system.py': 'System test suite'
    }
    
    all_exist = True
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"✅ {file:45} - {desc}")
        else:
            print(f"❌ {file:45} - MISSING!")
            all_exist = False
    
    return all_exist

def test_config_values():
    """Test configuration has proper limits"""
    print_header("TEST 2: Configuration Values")
    
    try:
        # Import config
        import config
        
        checks = [
            ('RIS_LIVE_MAX_MESSAGES', 'RIS Live message limit'),
            ('RIS_LIVE_MAX_RUNTIME_HOURS', 'RIS Live runtime limit'),
            ('TEST_DATA_MAX_RECORDS', 'Test data record limit'),
            ('DB_MAX_RECORDS', 'Database record limit'),
            ('DB_RETENTION_DAYS', 'Database retention period'),
            ('DB_AUTO_CLEANUP_ENABLED', 'Auto cleanup enabled'),
            ('MEMORY_LIMIT_MB', 'Memory limit'),
        ]
        
        print("Configuration Limits:")
        all_present = True
        for attr, desc in checks:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"  ✅ {desc:35} = {value}")
            else:
                print(f"  ❌ {desc:35} - NOT FOUND")
                all_present = False
        
        return all_present
        
    except ImportError as e:
        print(f"❌ Cannot import config.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False

def test_ris_live_limits():
    """Test RIS Live client has limits"""
    print_header("TEST 3: RIS Live Client Limits")
    
    file_path = 'routinator/ris_live_client (1).py'
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'max_messages': 'Message limit parameter',
        'max_runtime': 'Runtime limit parameter',
        'message_count': 'Message counter',
        'start_time': 'Start time tracking',
        'self.running = False': 'Graceful stop mechanism'
    }
    
    all_found = True
    for check, desc in checks.items():
        if check in content:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc} - NOT FOUND")
            all_found = False
    
    return all_found

def test_test_data_generator_limits():
    """Test data generator has limits"""
    print_header("TEST 4: Test Data Generator Limits")
    
    file_path = 'routinator/main (1).py'
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'max_records': 'Record limit parameter',
        'max_runtime': 'Runtime limit parameter',
        'record_count': 'Record counter',
        'stop_flag': 'Stop flag mechanism'
    }
    
    all_found = True
    for check, desc in checks.items():
        if check in content:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc} - NOT FOUND")
            all_found = False
    
    return all_found

def test_database_cleanup():
    """Test database cleanup utility"""
    print_header("TEST 5: Database Cleanup Utility")
    
    try:
        # Try to import the cleanup module
        sys.path.insert(0, os.getcwd())
        from database_cleanup import DatabaseCleanup
        
        print("✅ DatabaseCleanup class exists")
        
        # Check if it has required methods
        required_methods = [
            'cleanup_old_records',
            'enforce_size_limit',
            'vacuum_tables',
            'run_full_cleanup'
        ]
        
        cleanup = DatabaseCleanup()
        all_methods = True
        
        for method in required_methods:
            if hasattr(cleanup, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
                all_methods = False
        
        return all_methods
        
    except ImportError as e:
        print(f"❌ Cannot import database_cleanup: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_sql_triggers():
    """Test SQL triggers file"""
    print_header("TEST 6: SQL Triggers")
    
    file_path = 'database_triggers.sql'
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'CREATE TABLE IF NOT EXISTS system_config': 'Config table creation',
        'CREATE OR REPLACE FUNCTION cleanup_old_records': 'Cleanup function',
        'CREATE OR REPLACE FUNCTION enforce_record_limit': 'Limit enforcement function',
        'CREATE TRIGGER': 'Trigger creation',
        'CREATE OR REPLACE VIEW system_stats': 'Stats view',
        'CREATE OR REPLACE PROCEDURE manual_cleanup': 'Manual cleanup procedure'
    }
    
    all_found = True
    for check, desc in checks.items():
        if check in content:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc} - NOT FOUND")
            all_found = False
    
    return all_found

def test_documentation():
    """Test documentation exists"""
    print_header("TEST 7: Documentation")
    
    docs = {
        'README_COMPLETE.md': 'Complete usage guide',
        'COMPLETE_FIX_SUMMARY.md': 'Fix summary',
        'FIXES_APPLIED.md': 'Detailed fixes'
    }
    
    all_exist = True
    for doc, desc in docs.items():
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            print(f"✅ {doc:30} - {desc} ({size:,} bytes)")
        else:
            print(f"❌ {doc:30} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  🧪 BGP ANOMALY DETECTION - SIMPLE FIX VERIFICATION")
    print("  (No API server or database connection required)")
    print("="*70)
    
    tests = [
        ("Fixed Files Exist", test_files_exist),
        ("Configuration Values", test_config_values),
        ("RIS Live Client Limits", test_ris_live_limits),
        ("Test Data Generator Limits", test_test_data_generator_limits),
        ("Database Cleanup Utility", test_database_cleanup),
        ("SQL Triggers", test_sql_triggers),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Your fixes are properly installed.")
        print("\n🚀 Next Steps:")
        print("   1. Install database triggers:")
        print("      psql -U postgres -d bgp_monitor -f database_triggers.sql")
        print("   2. Start API server (in separate terminal):")
        print("      python api.py")
        print("   3. Run full system tests:")
        print("      python test_system.py")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Review the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
