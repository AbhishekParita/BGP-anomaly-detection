"""
Drift Monitor Dashboard
=======================
Monitor drift detection status and retraining activities.

Shows:
- Current drift status for each model
- Latest drift reports
- Retraining history
- Model versions and timestamps
"""

import os
import sys
import json
from datetime import datetime
from glob import glob
from collections import defaultdict


def print_header(title: str):
    """Print section header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def check_drift_flags():
    """Check for active retraining flags"""
    print_header("ACTIVE RETRAINING FLAGS")
    
    flags = {
        'LSTM': 'model_artifacts/retrain_lstm.flag',
        'Isolation Forest': 'model_artifacts/retrain_isolation_forest.flag',
        'Heuristic': 'model_artifacts/retrain_heuristic.flag'
    }
    
    active_flags = []
    
    for model, path in flags.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                flag_data = json.load(f)
            
            timestamp = flag_data.get('timestamp', 'Unknown')
            reason = flag_data.get('reason', 'Unknown')
            
            active_flags.append((model, timestamp, reason))
            print(f"🚩 {model:18s} Flag created: {timestamp}")
            print(f"   Reason: {reason}")
    
    if not active_flags:
        print("✅ No active retraining flags")
    
    return active_flags


def show_drift_reports():
    """Show latest drift reports"""
    print_header("LATEST DRIFT REPORTS (Last 5)")
    
    # Find all drift reports
    reports = glob("model_artifacts/drift_report_*.json")
    reports.sort(key=os.path.getmtime, reverse=True)
    
    if not reports:
        print("No drift reports found")
        return
    
    # Group by model
    by_model = defaultdict(list)
    for report in reports[:15]:  # Last 15 reports
        filename = os.path.basename(report)
        if 'lstm' in filename:
            by_model['LSTM'].append(report)
        elif 'isolation_forest' in filename:
            by_model['Isolation Forest'].append(report)
        elif 'heuristic' in filename:
            by_model['Heuristic'].append(report)
    
    # Show latest for each model
    for model in ['LSTM', 'Isolation Forest', 'Heuristic']:
        if model in by_model and by_model[model]:
            print(f"\n{model}:")
            
            for report_path in by_model[model][:2]:  # Latest 2
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                timestamp = report.get('timestamp', 'Unknown')
                drift_detected = report.get('drift_detected', False)
                
                status = "⚠️ DRIFT" if drift_detected else "✅ OK"
                print(f"  {status} {timestamp}")
                
                if drift_detected and 'metrics' in report:
                    metrics = report['metrics']
                    print(f"     Score shift: {metrics.get('score_shift', 0):.3f}")
                    print(f"     Anomaly rate change: {metrics.get('anomaly_rate_change', 0):.3f}")


def show_model_versions():
    """Show current model versions"""
    print_header("MODEL VERSIONS")
    
    models = {
        'LSTM': 'model_output/lstm_model_for_pkl.weights.h5',
        'Isolation Forest': 'model_artifacts/iso_forest_bgp_production.pkl',
        'Heuristic Rules': 'model_artifacts/heuristic_rules.json'
    }
    
    for model, path in models.items():
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            mod_time = datetime.fromtimestamp(mtime)
            age_hours = (datetime.now() - mod_time).total_seconds() / 3600
            size = os.path.getsize(path)
            
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.2f} MB"
            else:
                size_str = f"{size/1024:.2f} KB"
            
            print(f"{model:18s} Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_hours:.1f}h ago)")
            print(f"{' '*18} Size: {size_str}")
        else:
            print(f"{model:18s} ❌ Not found")


def show_retraining_history():
    """Show retraining history from backups"""
    print_header("RETRAINING HISTORY (Backups)")
    
    # Find backup files
    if_backups = glob("model_artifacts/iso_forest_bgp_production_backup_*.pkl")
    heuristic_backups = glob("model_artifacts/heuristic_rules_backup_*.json")
    lstm_backups = glob("model_output/lstm_model_for_pkl_backup_*.weights.h5")
    
    all_backups = [
        ('IF', b, os.path.getmtime(b)) for b in if_backups
    ] + [
        ('Heuristic', b, os.path.getmtime(b)) for b in heuristic_backups
    ] + [
        ('LSTM', b, os.path.getmtime(b)) for b in lstm_backups
    ]
    
    # Sort by time
    all_backups.sort(key=lambda x: x[2], reverse=True)
    
    if not all_backups:
        print("No retraining history found (no backups)")
        return
    
    print(f"Total retraining events: {len(all_backups)}")
    print(f"\nLast 5 retraining events:")
    
    for model, path, mtime in all_backups[:5]:
        mod_time = datetime.fromtimestamp(mtime)
        print(f"  {model:12s} {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")


def show_drift_monitor_status():
    """Show drift monitor service status"""
    print_header("DRIFT MONITOR SERVICE")
    
    log_path = "drift_monitor.log"
    
    if os.path.exists(log_path):
        # Get last few lines
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        if lines:
            last_line = lines[-1].strip()
            print(f"Last log entry: {last_line}")
            
            # Check age
            mtime = os.path.getmtime(log_path)
            mod_time = datetime.fromtimestamp(mtime)
            age_minutes = (datetime.now() - mod_time).total_seconds() / 60
            
            if age_minutes < 65:  # Less than 1 hour + buffer
                print(f"✅ Service active (last activity: {age_minutes:.1f} minutes ago)")
            else:
                print(f"⚠️ Service may be stopped (last activity: {age_minutes:.1f} minutes ago)")
    else:
        print("❌ No log file found (service not started)")
        print("   Start with: python services/drift_monitor.py")


def main():
    """Run dashboard"""
    print("\n" + "="*60)
    print("DRIFT MONITOR DASHBOARD")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show sections
    show_drift_monitor_status()
    check_drift_flags()
    show_drift_reports()
    show_model_versions()
    show_retraining_history()
    
    print("\n" + "="*60)
    print("Dashboard complete")
    print("="*60)
    print("\nCommands:")
    print("  Start monitoring: python services/drift_monitor.py")
    print("  Test retraining:  python test_retraining.py")
    print("  Manual retrain:   python retrain_<model>.py")


if __name__ == "__main__":
    main()
