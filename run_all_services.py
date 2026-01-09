"""
BGP Anomaly Detection System - Service Orchestrator

This script manages all 8 services in the detection pipeline:
1. RIS Live Collector (collects real BGP data)
2. Feature Aggregator (aggregates data into time windows)
3. Heuristic Detector (rule-based detection)
4. LSTM Detector (temporal pattern detection)
5. Isolation Forest Detector (statistical outlier detection)
6. Ensemble Coordinator (combines all detector scores)
7. Correlation Engine (correlates alerts and generates incidents)
8. Drift Monitor (monitors model performance and triggers retraining)

Usage:
    python run_all_services.py start    # Start all services
    python run_all_services.py stop     # Stop all services
    python run_all_services.py status   # Check service status
    python run_all_services.py restart  # Restart all services

Author: BGP Anomaly Detection System
Created: 2026-01-07
"""

import os
import sys
import time
import signal
import subprocess
import psutil
from pathlib import Path
from typing import List, Dict, Optional

# Service configuration
SERVICES = [
    {
        'name': 'RIS Live Collector',
        'script': 'services/ris_live_collector.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 5,  # Wait 5 seconds before starting next service
        'pid_file': '.ris_collector.pid'
    },
    {
        'name': 'Feature Aggregator',
        'script': 'services/feature_aggregator.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 3,
        'pid_file': '.feature_aggregator.pid'
    },
    {
        'name': 'Heuristic Detector',
        'script': 'services/heuristic_detector.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 2,
        'pid_file': '.heuristic_detector.pid'
    },
    {
        'name': 'LSTM Detector',
        'script': 'services/lstm_detector.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 2,
        'pid_file': '.lstm_detector.pid'
    },
    {
        'name': 'Isolation Forest Detector',
        'script': 'services/isolation_forest_detector.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 2,
        'pid_file': '.if_detector.pid'
    },
    {
        'name': 'Ensemble Coordinator',
        'script': 'services/ensemble_coordinator.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 2,
        'pid_file': '.ensemble_coordinator.pid'
    },
    {
        'name': 'Correlation Engine',
        'script': 'services/correlation_engine.py',
        'args': ['--mode', 'continuous'],
        'wait_after_start': 2,
        'pid_file': '.correlation_engine.pid'
    },
    {
        'name': 'Drift Monitor',
        'script': 'services/drift_monitor.py',
        'args': [],
        'wait_after_start': 0,
        'pid_file': '.drift_monitor.pid'

    }
]


class ServiceOrchestrator:
    """Manages lifecycle of all BGP anomaly detection services."""
    
    def __init__(self):
        """Initialize orchestrator."""
        self.base_dir = Path(__file__).parent
        self.python_exe = sys.executable
        self.processes: Dict[str, subprocess.Popen] = {}
    
    def start_service(self, service: Dict) -> bool:
        """
        Start a single service.
        
        Args:
            service: Service configuration dictionary
            
        Returns:
            bool: True if started successfully
        """
        try:
            script_path = self.base_dir / service['script']
            
            if not script_path.exists():
                print(f"[ERROR] Script not found: {script_path}")
                return False
            
            # Check if already running
            pid = self.get_service_pid(service['pid_file'])
            if pid and self.is_process_running(pid):
                print(f"[WARNING] {service['name']} already running (PID: {pid})")
                return True
            
            # Start the service
            cmd = [self.python_exe, str(script_path)] + service['args']
            
            print(f"[STARTING] {service['name']}...")
            
            # Use CREATE_NEW_PROCESS_GROUP on Windows to allow proper termination
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Save PID
            pid_file = self.base_dir / service['pid_file']
            pid_file.write_text(str(process.pid))
            
            # Store process
            self.processes[service['name']] = process
            
            # Wait to ensure service starts properly
            time.sleep(service['wait_after_start'])
            
            # Check if still running
            if process.poll() is None:
                print(f"[OK] {service['name']} started (PID: {process.pid})")
                return True
            else:
                print(f"[ERROR] {service['name']} failed to start")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to start {service['name']}: {e}")
            return False
    
    def stop_service(self, service: Dict) -> bool:
        """
        Stop a single service.
        
        Args:
            service: Service configuration dictionary
            
        Returns:
            bool: True if stopped successfully
        """
        try:
            pid_file = self.base_dir / service['pid_file']
            
            if not pid_file.exists():
                print(f"[WARNING] {service['name']} not running (no PID file)")
                return True
            
            pid = int(pid_file.read_text().strip())
            
            if not self.is_process_running(pid):
                print(f"[WARNING] {service['name']} not running (stale PID)")
                pid_file.unlink()
                return True
            
            print(f"[STOPPING] {service['name']} (PID: {pid})...")
            
            # Try graceful termination first
            try:
                process = psutil.Process(pid)
                process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                process.wait(timeout=5)
                print(f"[OK] {service['name']} stopped gracefully")
                
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown fails
                print(f"[WARNING] Force killing {service['name']}...")
                process.kill()
                process.wait(timeout=2)
                print(f"[OK] {service['name']} force killed")
            
            # Clean up PID file
            if pid_file.exists():
                pid_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to stop {service['name']}: {e}")
            return False
    
    def get_service_pid(self, pid_file: str) -> Optional[int]:
        """
        Get PID from PID file.
        
        Args:
            pid_file: Path to PID file
            
        Returns:
            int or None: PID if file exists and valid
        """
        try:
            pid_path = self.base_dir / pid_file
            if pid_path.exists():
                return int(pid_path.read_text().strip())
        except:
            pass
        return None
    
    def is_process_running(self, pid: int) -> bool:
        """
        Check if process is running.
        
        Args:
            pid: Process ID
            
        Returns:
            bool: True if process is running
        """
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False
    
    def get_service_status(self, service: Dict) -> Dict:
        """
        Get status of a service.
        
        Args:
            service: Service configuration dictionary
            
        Returns:
            dict: Status information
        """
        pid = self.get_service_pid(service['pid_file'])
        
        if pid and self.is_process_running(pid):
            try:
                process = psutil.Process(pid)
                return {
                    'name': service['name'],
                    'status': 'RUNNING',
                    'pid': pid,
                    'cpu_percent': process.cpu_percent(interval=0.1),
                    'memory_mb': process.memory_info().rss / 1024 / 1024
                }
            except:
                return {
                    'name': service['name'],
                    'status': 'UNKNOWN',
                    'pid': pid
                }
        else:
            return {
                'name': service['name'],
                'status': 'STOPPED',
                'pid': None
            }
    
    def start_all(self):
        """Start all services in order."""
        print("=" * 70)
        print("BGP ANOMALY DETECTION SYSTEM - STARTING ALL SERVICES")
        print("=" * 70)
        print()
        
        success_count = 0
        for service in SERVICES:
            if self.start_service(service):
                success_count += 1
            print()
        
        print("=" * 70)
        print(f"STARTUP COMPLETE: {success_count}/{len(SERVICES)} services started")
        print("=" * 70)
        print()
        print("To monitor services: python run_all_services.py status")
        print("To stop all services: python run_all_services.py stop")
        print()
    
    def stop_all(self):
        """Stop all services in reverse order."""
        print("=" * 70)
        print("BGP ANOMALY DETECTION SYSTEM - STOPPING ALL SERVICES")
        print("=" * 70)
        print()
        
        # Stop in reverse order
        success_count = 0
        for service in reversed(SERVICES):
            if self.stop_service(service):
                success_count += 1
            print()
        
        print("=" * 70)
        print(f"SHUTDOWN COMPLETE: {success_count}/{len(SERVICES)} services stopped")
        print("=" * 70)
    
    def show_status(self):
        """Show status of all services."""
        print("=" * 70)
        print("BGP ANOMALY DETECTION SYSTEM - SERVICE STATUS")
        print("=" * 70)
        print()
        
        running_count = 0
        
        for service in SERVICES:
            status = self.get_service_status(service)
            
            status_str = status['status']
            if status['status'] == 'RUNNING':
                running_count += 1
                print(f"[OK] {status['name']:<30} RUNNING")
                print(f"     PID: {status['pid']}, CPU: {status['cpu_percent']:.1f}%, RAM: {status['memory_mb']:.1f}MB")
            elif status['status'] == 'STOPPED':
                print(f"[STOPPED] {status['name']:<30} NOT RUNNING")
            else:
                print(f"[WARNING] {status['name']:<30} UNKNOWN")
            print()
        
        print("=" * 70)
        print(f"STATUS SUMMARY: {running_count}/{len(SERVICES)} services running")
        print("=" * 70)
    
    def restart_all(self):
        """Restart all services."""
        print("Restarting all services...")
        print()
        self.stop_all()
        print()
        time.sleep(2)
        print()
        self.start_all()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='BGP Anomaly Detection System - Service Orchestrator'
    )
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'status', 'restart'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    orchestrator = ServiceOrchestrator()
    
    if args.command == 'start':
        orchestrator.start_all()
    elif args.command == 'stop':
        orchestrator.stop_all()
    elif args.command == 'status':
        orchestrator.show_status()
    elif args.command == 'restart':
        orchestrator.restart_all()


if __name__ == "__main__":
    main()
