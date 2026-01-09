"""
Emergency Fix Script
Fixes all critical issues found in diagnosis
"""

import os
import sys
import subprocess
from pathlib import Path

def print_step(step, description):
    print(f"\n{'='*70}")
    print(f"STEP {step}: {description}")
    print('='*70)

def run_command(cmd, description):
    """Run a command and report result"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout:
                print(result.stdout[:500])  # First 500 chars
            return True
        else:
            print(f"❌ Failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_postgres():
    """Check if PostgreSQL is accessible using Python"""
    print_step(1, "Checking PostgreSQL")
    
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'bgp_monitor'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'anand'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL connected: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\n💡 Check .env file for correct credentials")
        return False

def init_database_schema():
    """Initialize database schema using Python (no psql needed)"""
    print_step(2, "Initialize Database Schema")
    
    # Use Python initializer (works without psql)
    init_script = Path('database/init_database.py')
    if init_script.exists():
        print(f"✅ Found Python initializer: {init_script}")
        print("   This will create all missing tables...")
        return run_command('python database/init_database.py', "Running Python initializer")
    else:
        print(f"❌ Database initializer not found: {init_script}")
        print("   Cannot create missing tables!")
        return False

def install_triggers():
    """Install database triggers using Python"""
    print_step(3, "Install Auto-Cleanup Triggers")
    
    # Use Python to execute SQL file
    print("✅ Installing triggers via Python...")
    return run_command('python install_triggers_python.py', "Installing triggers")

def cleanup_database():
    """Run database cleanup"""
    print_step(4, "Clean Up Database (6.4M records!)")
    
    cleanup_script = Path('database_cleanup.py')
    if cleanup_script.exists():
        print(f"✅ Found cleanup script: {cleanup_script}")
        print("⚠️  This will delete old records to get under limit")
        
        response = input("\n   Proceed with cleanup? (yes/no): ")
        if response.lower() == 'yes':
            return run_command('python database_cleanup.py', "Running cleanup")
        else:
            print("⏭️  Skipped cleanup")
            return True
    else:
        print(f"❌ Cleanup script not found: {cleanup_script}")
        return False

def train_isolation_forest():
    """Train Isolation Forest model"""
    print_step(5, "Train Isolation Forest Model")
    
    train_script = Path('train_if_model.py')
    if train_script.exists():
        print(f"✅ Found training script: {train_script}")
        print("⚠️  This may take several minutes")
        
        response = input("\n   Proceed with training? (yes/no): ")
        if response.lower() == 'yes':
            return run_command('python train_if_model.py', "Training Isolation Forest")
        else:
            print("⏭️  Skipped training")
            return True
    else:
        print(f"❌ Training script not found: {train_script}")
        return False

def list_obsolete_files():
    """List obsolete files for manual removal"""
    print_step(6, "Obsolete Files to Remove")
    
    obsolete = [
        'kafka_2.13-3.9.0/',
        'stream_generator.py',
        'hybrid_detector.py',
        'bmp_generator.py',
        'kafka_2.13-3.9.0.tgz',
        'kafka_2.13-3.9.0.tgz.1'
    ]
    
    found = []
    for item in obsolete:
        path = Path(item)
        if path.exists():
            found.append(item)
            size = "DIR" if path.is_dir() else f"{path.stat().st_size:,} bytes"
            print(f"   ⚠️  {item} - {size}")
    
    if found:
        print(f"\n   Found {len(found)} obsolete items")
        print("\n   To remove them, run:")
        print("   # For files:")
        for item in found:
            if not item.endswith('/'):
                print(f"   Remove-Item '{item}'")
        print("\n   # For directories:")
        for item in found:
            if item.endswith('/'):
                print(f"   Remove-Item -Recurse -Force '{item}'")
    else:
        print("   ✅ No obsolete files found")

def main():
    """Run all fixes"""
    print("\n" + "="*70)
    print("  EMERGENCY FIX SCRIPT")
    print("  Fixing Critical Issues from Diagnosis")
    print("="*70)
    
    # Check PostgreSQL first
    postgres_ok = check_postgres()
    
    if postgres_ok:
        # Database fixes (all use Python - no psql needed!)
        init_database_schema()
        install_triggers()
        cleanup_database()
    else:
        print("\n⚠️  Skipping database operations (PostgreSQL not accessible)")
        print("   Check your .env file and PostgreSQL service")
        print("   Required: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT")
    
    # Model training
    train_isolation_forest()
    
    # List obsolete files
    list_obsolete_files()
    
    print("\n" + "="*70)
    print("  FIX SCRIPT COMPLETE")
    print("="*70)
    print("\n  Next Steps:")
    print("  1. Run diagnosis again: python system_diagnosis.py")
    print("  2. Check limits are working in RIS Live client")
    print("  3. Remove obsolete Kafka files (see list above)")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Fix script interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
