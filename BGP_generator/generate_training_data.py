#!/usr/bin/env python3
"""
Generate comprehensive training data for model training
Simulates 30 days of BGP behavior patterns
"""

import subprocess
import time
import random
from datetime import datetime, timedelta

def run_bmp_generator(mode, duration_seconds, **kwargs):
    """Run BMP generator for specified duration"""
    cmd = [
        'python', 'bmp_generator.py',
        '--host', 'localhost',
        '--port', '5000',
        '--mode', mode
    ]
    
    for key, value in kwargs.items():
        flag_name = f'--{key.replace("_", "-")}'
        
        # 2. FIX: Only add the flag for boolean True values (toggles)
        if isinstance(value, bool):
            if value:
                cmd.append(flag_name)
        # 3. Handle all other non-boolean values (integers, strings, etc.)
        else:
            cmd.extend([flag_name, str(value)])
    
    #for key, value in kwargs.items():
    #    cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    print(f"[{datetime.now()}] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    time.sleep(duration_seconds)
    proc.terminate()
    proc.wait()

def simulate_normal_operations(hours=24):
    """Simulate normal BGP operations"""
    print(f"\n{'='*80}")
    print(f"SIMULATING NORMAL OPERATIONS ({hours} hours)")
    print(f"{'='*80}\n")
    
    # Normal route updates every 10 seconds
    run_bmp_generator(
        mode='routes',
        duration_seconds=hours * 60,  # Compressed time
        routers=3,
        peers_per_router=2,
        continuous=True,
        interval=10,
        full_attributes=True
    )

def simulate_route_flapping(count=50):
    """Simulate route flapping for dampening training"""
    print(f"\n{'='*80}")
    print(f"SIMULATING ROUTE FLAPPING ({count} flaps)")
    print(f"{'='*80}\n")
    
    for i in range(count):
        # Announce
        run_bmp_generator(mode='routes', duration_seconds=2,
                         routers=1, peers_per_router=1)
        time.sleep(1)
        
        # Withdraw
        run_bmp_generator(mode='peer-down', duration_seconds=1,
                         routers=1, peers_per_router=1)
        time.sleep(1)
        
        print(f"Flap {i+1}/{count} complete")

def simulate_peer_instability():
    """Simulate peer down/up cycles"""
    print(f"\n{'='*80}")
    print(f"SIMULATING PEER INSTABILITY")
    print(f"{'='*80}\n")
    
    for i in range(10):
        # Peer up
        run_bmp_generator(mode='peer-up', duration_seconds=30,
                         routers=2, peers_per_router=2)
        
        # Stable period
        run_bmp_generator(mode='routes', duration_seconds=60,
                         routers=2, peers_per_router=2,
                         continuous=True, interval=5)
        
        # Peer down
        run_bmp_generator(mode='peer-down', duration_seconds=5,
                         routers=2, peers_per_router=2)
        
        time.sleep(5)

def simulate_churn_events():
    """Simulate high churn periods"""
    print(f"\n{'='*80}")
    print(f"SIMULATING HIGH CHURN EVENTS")
    print(f"{'='*80}\n")
    
    # Sudden burst of updates
    run_bmp_generator(
        mode='routes',
        duration_seconds=300,  # 5 minutes of high activity
        routers=3,
        peers_per_router=3,
        continuous=True,
        interval=1,  # Very frequent updates
        full_attributes=True,
        route_count=50
    )

def main():
    """Generate complete training dataset"""
    print("="*80)
    print("BGP TRAINING DATA GENERATOR")
    print("Generating 30 days of simulated BGP data (compressed)")
    print("="*80)
    
    # Day 1-7: Normal operations
    simulate_normal_operations(hours=168)  # 7 days compressed to ~3 hours
    
    # Day 8: Route flapping events
    simulate_route_flapping(count=50)
    
    # Day 9-10: Peer instability
    simulate_peer_instability()
    
    # Day 11: High churn period
    simulate_churn_events()
    
    # Day 12-30: More normal operations with occasional anomalies
    for day in range(19):
        print(f"\nDay {12+day}: Normal operations with random events")
        simulate_normal_operations(hours=20)
        
        # Random anomaly
        if random.random() < 0.3:
            if random.random() < 0.5:
                simulate_route_flapping(count=10)
            else:
                simulate_churn_events()
    
    print("\n" + "="*80)
    print("TRAINING DATA GENERATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Export data from PostgreSQL")
    print("2. Train models with exported data")
    print("3. Validate model performance")

if __name__ == "__main__":
    main()