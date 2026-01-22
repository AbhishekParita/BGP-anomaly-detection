"""
Manual Retraining Approval CLI Tool
====================================
Interactive tool to approve and start model retraining when drift is detected.

Usage:
    python approve_retraining.py               # Check all models
    python approve_retraining.py lstm          # Approve specific model
    python approve_retraining.py --list        # List pending retraining
    python approve_retraining.py --status      # Show drift status
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
from typing import Optional, List, Dict
import psycopg2
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class RetrainingApprovalTool:
    """CLI tool for manual retraining approval"""
    
    def __init__(self):
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        self.model_scripts = {
            'lstm': 'retrain_lstm.py',
            'isolation_forest': 'retrain_isolation_forest.py',
            'heuristic': 'retrain_heuristic.py'
        }
        
        self.conn = None
        self.cursor = None
    
    def connect_db(self) -> bool:
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def get_drift_status(self, model_name: Optional[str] = None) -> List[Dict]:
        """Get drift status from database"""
        try:
            if model_name:
                self.cursor.execute("""
                    SELECT model_name, drift_detected, detection_timestamp, 
                           status, message, drift_metrics
                    FROM drift_status
                    WHERE model_name = %s AND drift_detected = TRUE;
                """, (model_name,))
            else:
                self.cursor.execute("""
                    SELECT model_name, drift_detected, detection_timestamp, 
                           status, message, drift_metrics
                    FROM drift_status
                    WHERE drift_detected = TRUE AND status = 'pending_approval';
                """)
            
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'model_name': row[0],
                    'drift_detected': row[1],
                    'detection_timestamp': row[2],
                    'status': row[3],
                    'message': row[4],
                    'drift_metrics': json.loads(row[5]) if row[5] else {}
                })
            
            return results
        except Exception as e:
            logger.error(f"Failed to get drift status: {e}")
            return []
    
    def get_drift_status_from_file(self, model_name: str) -> Optional[Dict]:
        """Get drift status from JSON file (fallback)"""
        status_file = f"model_artifacts/drift_status_{model_name}.json"
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def list_pending_retraining(self):
        """List all models with pending retraining"""
        print("\n" + "="*70)
        print("PENDING RETRAINING APPROVALS")
        print("="*70)
        
        pending = self.get_drift_status()
        
        if not pending:
            print("\n✅ No pending retraining approvals")
            print("   All models are healthy!")
            return
        
        for i, status in enumerate(pending, 1):
            print(f"\n[{i}] Model: {status['model_name'].upper()}")
            print(f"    Status: {status['status']}")
            print(f"    Message: {status['message']}")
            print(f"    Detected: {status['detection_timestamp']}")
            
            metrics = status.get('drift_metrics', {})
            if metrics:
                print(f"    Drift Metrics:")
                for key, value in metrics.items():
                    print(f"      - {key}: {value:.4f}")
        
        print("\n" + "="*70)
    
    def show_drift_status(self):
        """Show comprehensive drift status for all models"""
        print("\n" + "="*70)
        print("DRIFT STATUS - ALL MODELS")
        print("="*70)
        
        models = ['lstm', 'isolation_forest', 'heuristic']
        
        for model in models:
            print(f"\n🤖 {model.upper().replace('_', ' ')}")
            print("-" * 70)
            
            # Check database
            status_list = self.get_drift_status(model)
            
            if status_list:
                status = status_list[0]
                print(f"   Status: ⚠️  {status['status']}")
                print(f"   Message: {status['message']}")
                print(f"   Detected: {status['detection_timestamp']}")
            else:
                # Check file
                file_status = self.get_drift_status_from_file(model)
                if file_status and file_status.get('drift_detected'):
                    print(f"   Status: ⚠️  {file_status['status']}")
                    print(f"   Message: {file_status['message']}")
                else:
                    print(f"   Status: ✅  OK - No drift detected")
        
        print("\n" + "="*70)
    
    def approve_and_start_retraining(self, model_name: str):
        """Approve and start retraining for a model"""
        print("\n" + "="*70)
        print(f"RETRAINING APPROVAL - {model_name.upper()}")
        print("="*70)
        
        # Check drift status
        status_list = self.get_drift_status(model_name)
        
        if not status_list:
            print(f"\n❌ No drift detected for {model_name}")
            print("   Retraining not required")
            return False
        
        status = status_list[0]
        
        # Show drift information
        print(f"\nDrift Details:")
        print(f"  Message: {status['message']}")
        print(f"  Detected: {status['detection_timestamp']}")
        
        metrics = status.get('drift_metrics', {})
        if metrics:
            print(f"  Drift Metrics:")
            for key, value in metrics.items():
                print(f"    - {key}: {value:.4f}")
        
        # Ask for confirmation
        print(f"\n⚠️  This will start retraining for {model_name}")
        print("   Training time estimate:")
        estimates = {
            'lstm': '20-30 minutes',
            'isolation_forest': '1-2 minutes',
            'heuristic': '30 seconds'
        }
        print(f"   - {estimates.get(model_name, '5-10 minutes')}")
        
        confirm = input("\n👉 Start retraining? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            print("\n❌ Retraining cancelled")
            return False
        
        # Update database status
        try:
            self.cursor.execute("""
                UPDATE drift_status
                SET status = 'retraining_in_progress',
                    retraining_approved = TRUE,
                    retraining_started_at = %s,
                    message = 'Retraining in progress...'
                WHERE model_name = %s;
            """, (datetime.now(), model_name))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Could not update database: {e}")
        
        # Start retraining script
        script_path = self.model_scripts.get(model_name)
        
        if not script_path:
            print(f"\n❌ Unknown model: {model_name}")
            return False
        
        print(f"\n🚀 Starting retraining script: {script_path}")
        print("   Please wait... This may take several minutes")
        print("   You can close this terminal - retraining will continue in background")
        
        try:
            # Run retraining script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print("\n" + "="*70)
                print("✅ RETRAINING COMPLETED SUCCESSFULLY")
                print("="*70)
                print("\nOutput:")
                print(result.stdout)
                return True
            else:
                print("\n" + "="*70)
                print("❌ RETRAINING FAILED")
                print("="*70)
                print("\nError:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("\n⚠️  Retraining is taking longer than expected")
            print("   Check logs for progress")
            return False
        except Exception as e:
            print(f"\n❌ Failed to start retraining: {e}")
            return False
    
    def interactive_mode(self):
        """Interactive mode for selecting and approving retraining"""
        while True:
            print("\n" + "="*70)
            print("RETRAINING APPROVAL TOOL - INTERACTIVE MODE")
            print("="*70)
            print("\nOptions:")
            print("  1. List pending retraining")
            print("  2. Show drift status (all models)")
            print("  3. Approve LSTM retraining")
            print("  4. Approve Isolation Forest retraining")
            print("  5. Approve Heuristic retraining")
            print("  6. Exit")
            
            choice = input("\n👉 Select option (1-6): ").strip()
            
            if choice == '1':
                self.list_pending_retraining()
            elif choice == '2':
                self.show_drift_status()
            elif choice == '3':
                self.approve_and_start_retraining('lstm')
            elif choice == '4':
                self.approve_and_start_retraining('isolation_forest')
            elif choice == '5':
                self.approve_and_start_retraining('heuristic')
            elif choice == '6':
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid option")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Retraining Approval Tool')
    parser.add_argument('model', nargs='?', choices=['lstm', 'isolation_forest', 'heuristic'],
                       help='Model to retrain (optional)')
    parser.add_argument('--list', action='store_true', help='List pending retraining')
    parser.add_argument('--status', action='store_true', help='Show drift status')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    tool = RetrainingApprovalTool()
    
    if not tool.connect_db():
        print("⚠️  Could not connect to database - using file-based status")
    
    try:
        if args.list:
            tool.list_pending_retraining()
        elif args.status:
            tool.show_drift_status()
        elif args.model:
            tool.approve_and_start_retraining(args.model)
        elif args.interactive:
            tool.interactive_mode()
        else:
            # Default: Interactive mode
            tool.interactive_mode()
    finally:
        tool.close()


if __name__ == "__main__":
    main()
