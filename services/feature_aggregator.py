"""
Feature Aggregator Service

This service aggregates raw BGP data into 1-minute time windows with statistical features.
It reads from raw_bgp_data table and writes to features table for ML model consumption.

Architecture Flow:
raw_bgp_data (raw BGP updates) → Feature Aggregator → features (1-min aggregated)

Features Computed (per 1-minute window):
1. total_updates: COUNT(*) - Total BGP updates
2. total_announcements: SUM(announcements)
3. total_withdrawals: SUM(withdrawals)
4. avg_path_length: AVG(path_length)
5. max_path_length: MAX(path_length)
6. unique_prefixes: COUNT(DISTINCT prefix)
7. unique_peers: COUNT(DISTINCT peer_addr)
8. avg_updates_per_peer: total_updates / unique_peers
9. is_anomaly: 0 (default, will be updated by ML models)

Author: BGP Anomaly Detection System
Created: 2026-01-07
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_aggregator.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FeatureAggregator:
    """
    Aggregates raw BGP data into time-series features for ML models.
    
    This service:
    1. Reads raw BGP data from raw_bgp_data table
    2. Aggregates data into 1-minute time windows
    3. Computes 9 statistical features per window
    4. Writes aggregated features to features table
    5. Tracks last processed timestamp to avoid reprocessing
    """
    
    def __init__(self):
        """Initialize Feature Aggregator with database connection."""
        self.db_conn = None
        self.db_cursor = None
        
        # Database configuration
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Aggregation settings (optimized for real-time)
        self.aggregation_interval = 30  # seconds (30 seconds for faster updates)
        self.processing_delay = 5  # seconds delay (reduced for real-time)
        self.batch_size = 50  # Process 50 time windows per batch (faster)
        
        # Statistics
        self.stats = {
            'windows_processed': 0,
            'features_inserted': 0,
            'errors': 0,
            'last_processed_time': None
        }
    
    def connect_db(self):
        """Establish database connection."""
        try:
            self.db_conn = psycopg2.connect(**self.db_config)
            self.db_cursor = self.db_conn.cursor()
            logger.info("[OK] Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Database connection failed: {e}")
            return False
    
    def close_db(self):
        """Close database connection."""
        if self.db_cursor:
            self.db_cursor.close()
        if self.db_conn:
            self.db_conn.close()
        logger.info("Database connection closed")
    
    def get_last_processed_timestamp(self) -> Optional[datetime]:
        """
        Get the last processed timestamp from features table.
        
        Returns:
            datetime: Last processed timestamp, or None if table is empty
        """
        try:
            query = """
            SELECT MAX(timestamp) 
            FROM features;
            """
            self.db_cursor.execute(query)
            result = self.db_cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            else:
                # If no features exist, start from the earliest raw data
                query = """
                SELECT MIN(timestamp) 
                FROM raw_bgp_data;
                """
                self.db_cursor.execute(query)
                result = self.db_cursor.fetchone()
                return result[0] if result and result[0] else None
                
        except Exception as e:
            logger.error(f"Error getting last processed timestamp: {e}")
            return None
    
    def aggregate_time_window(self, start_time: datetime, end_time: datetime) -> Optional[Dict]:
        """
        Aggregate raw BGP data for a specific time window.
        
        Args:
            start_time: Start of the time window
            end_time: End of the time window
            
        Returns:
            dict: Aggregated features, or None if no data in window
        """
        try:
            query = """
            SELECT 
                COUNT(*) as total_updates,
                SUM(announcements) as total_announcements,
                SUM(withdrawals) as total_withdrawals,
                AVG(path_length) as avg_path_length,
                MAX(path_length) as max_path_length,
                COUNT(DISTINCT prefix) as unique_prefixes,
                COUNT(DISTINCT peer_addr) as unique_peers
            FROM raw_bgp_data
            WHERE timestamp >= %s AND timestamp < %s;
            """
            
            self.db_cursor.execute(query, (start_time, end_time))
            result = self.db_cursor.fetchone()
            
            if result and result[0] > 0:  # If there are updates in this window
                total_updates, total_ann, total_with, avg_path, max_path, unique_prefixes, unique_peers = result
                
                # Calculate derived features
                avg_updates_per_peer = total_updates / unique_peers if unique_peers > 0 else 0
                
                return {
                    'timestamp': start_time,
                    'total_updates': total_updates,
                    'total_announcements': total_ann or 0,
                    'total_withdrawals': total_with or 0,
                    'avg_path_length': float(avg_path) if avg_path else 0.0,
                    'max_path_length': max_path or 0,
                    'unique_prefixes': unique_prefixes or 0,
                    'unique_peers': unique_peers or 0,
                    'avg_updates_per_peer': float(avg_updates_per_peer)
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error aggregating time window {start_time}: {e}")
            self.stats['errors'] += 1
            return None
    
    def insert_features(self, features_list: List[Dict]) -> int:
        """
        Batch insert aggregated features into features table.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            int: Number of records inserted
        """
        if not features_list:
            return 0
            
        try:
            query = """
            INSERT INTO features (
                timestamp, peer_addr, total_updates, announcements, withdrawals,
                path_length, unique_peers
            ) VALUES %s;
            """
            
            # Prepare values for batch insert (use 'all' for peer_addr as aggregate)
            values = [
                (
                    f['timestamp'], 'aggregate', f['total_updates'], 
                    f['total_announcements'], f['total_withdrawals'],
                    f['avg_path_length'], f['unique_peers']
                )
                for f in features_list
            ]
            
            execute_values(self.db_cursor, query, values)
            self.db_conn.commit()
            
            inserted_count = len(features_list)
            self.stats['features_inserted'] += inserted_count
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            self.db_conn.rollback()
            self.stats['errors'] += 1
            return 0
    
    def process_pending_windows(self) -> int:
        """
        Process all pending time windows since last processing.
        
        Returns:
            int: Number of windows processed
        """
        try:
            # Get last processed timestamp
            last_processed = self.get_last_processed_timestamp()
            
            if last_processed is None:
                logger.warning("No data to process yet")
                return 0
            
            # Get current time minus processing delay
            current_time = datetime.now() - timedelta(seconds=self.processing_delay)
            
            # Truncate to minute boundary
            current_time = current_time.replace(second=0, microsecond=0)
            last_processed = last_processed.replace(second=0, microsecond=0)
            
            # If we've already processed up to current time, nothing to do
            if last_processed >= current_time:
                logger.debug(f"Already up-to-date (last: {last_processed}, current: {current_time})")
                return 0
            
            logger.info(f"Processing windows from {last_processed} to {current_time}")
            
            # Process windows in batches
            features_batch = []
            windows_count = 0
            
            window_start = last_processed
            while window_start < current_time:
                window_end = window_start + timedelta(seconds=self.aggregation_interval)
                
                # Aggregate this time window
                features = self.aggregate_time_window(window_start, window_end)
                
                if features:
                    features_batch.append(features)
                    windows_count += 1
                    
                    # Insert batch when it reaches batch_size
                    if len(features_batch) >= self.batch_size:
                        inserted = self.insert_features(features_batch)
                        logger.info(f"[OK] Inserted {inserted} feature windows | Total: {self.stats['features_inserted']}")
                        features_batch = []
                
                # Move to next window
                window_start = window_end
            
            # Insert remaining features
            if features_batch:
                inserted = self.insert_features(features_batch)
                logger.info(f"[OK] Inserted {inserted} feature windows | Total: {self.stats['features_inserted']}")
            
            self.stats['windows_processed'] += windows_count
            self.stats['last_processed_time'] = current_time
            
            # Log statistics every 10 windows
            if windows_count > 0 and self.stats['windows_processed'] % 10 == 0:
                logger.info(f"[STATS] {self.stats['windows_processed']} windows | "
                          f"{self.stats['features_inserted']} features | "
                          f"{self.stats['errors']} errors")
            
            return windows_count
            
        except Exception as e:
            logger.error(f"Error processing pending windows: {e}")
            self.stats['errors'] += 1
            return 0
    
    async def run_continuous(self):
        """
        Run feature aggregation continuously in a loop.
        """
        logger.info("=" * 60)
        logger.info("FEATURE AGGREGATOR SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"Aggregation interval: {self.aggregation_interval} seconds")
        logger.info(f"Processing delay: {self.processing_delay} seconds")
        logger.info(f"Batch size: {self.batch_size} windows")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Connect to database
        if not self.connect_db():
            logger.error("Cannot start service without database connection")
            return
        
        try:
            while True:
                # Process pending windows
                processed = self.process_pending_windows()
                
                if processed > 0:
                    logger.info(f"Processed {processed} time windows in this cycle")
                
                # Wait before next cycle (check every 30 seconds)
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("\n[WARNING] Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Print final statistics
            self.close_db()
            logger.info("=" * 60)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total windows processed: {self.stats['windows_processed']}")
            logger.info(f"Features inserted: {self.stats['features_inserted']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info(f"Last processed time: {self.stats['last_processed_time']}")
            logger.info("=" * 60)
    
    async def run_once(self):
        """
        Run feature aggregation once (for testing/batch processing).
        """
        logger.info("Running feature aggregation (single pass)...")
        
        if not self.connect_db():
            logger.error("Cannot run without database connection")
            return
        
        try:
            processed = self.process_pending_windows()
            logger.info(f"[OK] Processed {processed} time windows")
        finally:
            self.close_db()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BGP Feature Aggregator Service')
    parser.add_argument(
        '--mode',
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous (daemon) or once (batch)'
    )
    
    args = parser.parse_args()
    
    aggregator = FeatureAggregator()
    
    if args.mode == 'continuous':
        asyncio.run(aggregator.run_continuous())
    else:
        asyncio.run(aggregator.run_once())


if __name__ == "__main__":
    main()
