#!/usr/bin/env python3
"""
Fast Training Data Generator for BGP Anomaly Detection
Generates compressed historical data by manipulating timestamps
CAUTION: This bypasses real-time collection - use only for initial training
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import random
import ipaddress
from typing import List, Dict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class SyntheticTrainingDataGenerator:
    """Generate synthetic BGP training data with realistic patterns"""
    
    def __init__(self, num_peers=5, seed=42):
        self.num_peers = num_peers
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate peer configurations
        self.peers = self._generate_peers()
    
    def _generate_peers(self) -> List[Dict]:
        """Generate peer configurations"""
        peers = []
        for i in range(self.num_peers):
            peer = {
                'peer_addr': f"192.168.{i+1}.1",
                'peer_as': 65000 + i,
                'peer_bgp_id': f"10.0.{i}.1",
                'region': random.choice(['us-east', 'us-west', 'eu-west', 'ap-south'])
            }
            peers.append(peer)
        return peers
    
    def generate_hourly_data(self, start_date: datetime, hours: int) -> pd.DataFrame:
        """
        Generate hourly BGP data for training
        
        Args:
            start_date: Starting timestamp
            hours: Number of hours to generate
        
        Returns:
            DataFrame with synthetic training data
        """
        print(f"Generating {hours} hours ({hours//24} days) of training data...")
        print(f"Start: {start_date}")
        print(f"End: {start_date + timedelta(hours=hours)}")
        
        all_data = []
        
        for hour in range(hours):
            timestamp = start_date + timedelta(hours=hour)
            
            # Generate data for each peer
            for peer in self.peers:
                record = self._generate_hour_record(peer, timestamp, hour)
                all_data.append(record)
            
            # Progress indicator
            if (hour + 1) % 168 == 0:  # Every week
                days_done = (hour + 1) // 24
                print(f"  Generated {days_done} days ({hour + 1} hours)...")
        
        df = pd.DataFrame(all_data)
        print(f"✓ Generated {len(df)} records for {self.num_peers} peers")
        
        return df
    
    def _generate_hour_record(self, peer: Dict, timestamp: datetime, hour_idx: int) -> Dict:
        """Generate a single hour's BGP data for a peer"""
        
        # Base activity level (with daily and weekly patterns)
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Daily pattern: lower activity at night (local time simulation)
        daily_factor = 0.5 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        # Weekly pattern: lower on weekends
        weekly_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Random variation
        noise = np.random.uniform(0.8, 1.2)
        
        activity_multiplier = daily_factor * weekly_factor * noise
        
        # Base metrics (normal operation)
        base_announcements = 50
        base_withdrawals = 10
        
        # Apply patterns
        announcements = max(1, int(base_announcements * activity_multiplier))
        withdrawals = max(0, int(base_withdrawals * activity_multiplier))
        
        # Derived metrics
        total_updates = announcements + withdrawals
        withdrawal_ratio = withdrawals / announcements if announcements > 0 else 0
        
        # Occasionally add small anomalies (5% of time) to make data realistic
        if random.random() < 0.05:
            # Small spike
            announcements = int(announcements * random.uniform(1.5, 2.5))
            withdrawals = int(withdrawals * random.uniform(1.5, 2.5))
            total_updates = announcements + withdrawals
            withdrawal_ratio = withdrawals / announcements if announcements > 0 else 0
        
        # Flapping (usually rare)
        estimated_flaps = min(withdrawals, announcements) * random.uniform(0.1, 0.3)
        
        # Path metrics
        avg_path_length = random.uniform(3.0, 5.0)
        unique_prefixes = int(announcements * random.uniform(0.7, 1.0))
        unique_paths = int(unique_prefixes * random.uniform(0.3, 0.6))
        unique_nexthops = random.randint(1, 3)
        
        # Message rate (updates per minute)
        message_rate = total_updates * 60
        
        return {
            'timestamp': timestamp,
            'peer_addr': peer['peer_addr'],
            'peer_as': peer['peer_as'],
            'region': peer['region'],
            'announcements': announcements,
            'withdrawals': withdrawals,
            'total_updates': total_updates,
            'unique_prefixes': unique_prefixes,
            'avg_path_length': avg_path_length,
            'unique_paths': unique_paths,
            'unique_nexthops': unique_nexthops,
            'withdrawal_ratio': withdrawal_ratio,
            'churn_rate': float(total_updates),
            'estimated_flaps': estimated_flaps,
            'message_rate': message_rate,
            'session_resets': 0
        }
    
    def add_anomalies(self, df: pd.DataFrame, anomaly_rate=0.02) -> pd.DataFrame:
        """
        Add synthetic anomalies to training data (optional)
        
        Args:
            df: Clean training data
            anomaly_rate: Fraction of data to mark as anomalous
        """
        print(f"\nAdding synthetic anomalies ({anomaly_rate*100:.1f}% of data)...")
        
        df = df.copy()
        num_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice(['route_leak', 'flapping', 'hijack', 'withdrawal_spike'])
            
            if anomaly_type == 'route_leak':
                # Massive announcement spike
                df.loc[idx, 'announcements'] *= random.uniform(10, 50)
                df.loc[idx, 'unique_prefixes'] *= random.uniform(10, 50)
                df.loc[idx, 'total_updates'] = df.loc[idx, 'announcements'] + df.loc[idx, 'withdrawals']
                
            elif anomaly_type == 'flapping':
                # High withdrawal/announcement ratio with many flaps
                base = df.loc[idx, 'announcements']
                df.loc[idx, 'withdrawals'] = base * random.uniform(2, 5)
                df.loc[idx, 'estimated_flaps'] = base * random.uniform(0.8, 1.5)
                df.loc[idx, 'total_updates'] = df.loc[idx, 'announcements'] + df.loc[idx, 'withdrawals']
                df.loc[idx, 'withdrawal_ratio'] = df.loc[idx, 'withdrawals'] / df.loc[idx, 'announcements']
                
            elif anomaly_type == 'hijack':
                # Sudden path length change + announcement
                df.loc[idx, 'avg_path_length'] = random.uniform(1.5, 2.5)
                df.loc[idx, 'announcements'] *= random.uniform(5, 15)
                df.loc[idx, 'total_updates'] = df.loc[idx, 'announcements'] + df.loc[idx, 'withdrawals']
                
            elif anomaly_type == 'withdrawal_spike':
                # Mass withdrawal event
                df.loc[idx, 'withdrawals'] *= random.uniform(20, 100)
                df.loc[idx, 'total_updates'] = df.loc[idx, 'announcements'] + df.loc[idx, 'withdrawals']
                df.loc[idx, 'withdrawal_ratio'] = df.loc[idx, 'withdrawals'] / max(1, df.loc[idx, 'announcements'])
            
            # Update derived metrics
            df.loc[idx, 'churn_rate'] = float(df.loc[idx, 'total_updates'])
            df.loc[idx, 'message_rate'] = df.loc[idx, 'total_updates'] * 60
        
        print(f"✓ Added {num_anomalies} synthetic anomalies")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """Save generated data to CSV"""
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved training data to: {output_path}")
        print(f"  Records: {len(df)}")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Peers: {df['peer_addr'].nunique()}")
    
    def display_summary(self, df: pd.DataFrame):
        """Display summary statistics"""
        print("\n" + "="*80)
        print("TRAINING DATA SUMMARY")
        print("="*80)
        print(f"Total records: {len(df)}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        print(f"Peers: {df['peer_addr'].nunique()}")
        print(f"\nKey Metrics (mean ± std):")
        print(f"  Announcements:     {df['announcements'].mean():8.1f} ± {df['announcements'].std():.1f}")
        print(f"  Withdrawals:       {df['withdrawals'].mean():8.1f} ± {df['withdrawals'].std():.1f}")
        print(f"  Total Updates:     {df['total_updates'].mean():8.1f} ± {df['total_updates'].std():.1f}")
        print(f"  Withdrawal Ratio:  {df['withdrawal_ratio'].mean():8.3f} ± {df['withdrawal_ratio'].std():.3f}")
        print(f"  Unique Prefixes:   {df['unique_prefixes'].mean():8.1f} ± {df['unique_prefixes'].std():.1f}")
        print(f"  Avg Path Length:   {df['avg_path_length'].mean():8.2f} ± {df['avg_path_length'].std():.2f}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Fast Training Data Generator for BGP Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 30 days of clean training data
  %(prog)s --days 30 --peers 5 --output training_data_30d.csv
  
  # Generate 7 days with synthetic anomalies
  %(prog)s --days 7 --peers 10 --anomalies 0.02 --output training_with_anomalies.csv
  
  # Generate 90 days for comprehensive training
  %(prog)s --days 90 --peers 8 --output training_data_90d.csv
  
  # Then train models:
  python train_models.py --csv-file training_data_30d.csv --output ./models
        """
    )
    
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of data to generate')
    parser.add_argument('--peers', type=int, default=5,
                       help='Number of BGP peers to simulate')
    parser.add_argument('--output', default='training_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--anomalies', type=float, default=0.0,
                       help='Fraction of data to mark as anomalous (0.02 = 2%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--start-date', 
                       help='Start date (YYYY-MM-DD), default: 30 days ago')
    
    args = parser.parse_args()
    
    # Determine start date
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        # Start from N days ago to simulate historical data
        start_date = datetime.now() - timedelta(days=args.days)
    
    # Create generator
    generator = SyntheticTrainingDataGenerator(
        num_peers=args.peers,
        seed=args.seed
    )
    
    # Generate data
    hours = args.days * 24
    df = generator.generate_hourly_data(start_date, hours)
    
    # Add anomalies if requested
    if args.anomalies > 0:
        df = generator.add_anomalies(df, anomaly_rate=args.anomalies)
    
    # Display summary
    generator.display_summary(df)
    
    # Save to CSV
    generator.save_to_csv(df, args.output)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"Train models using this data:")
    print(f"  python train_models.py --csv-file {args.output} --output ./trained_models")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())