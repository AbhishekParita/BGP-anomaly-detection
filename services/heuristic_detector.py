"""
Heuristic Detector Service

This service runs rule-based anomaly detection on aggregated BGP features.
It uses domain knowledge and BGP best practices to identify suspicious patterns.

Architecture Flow:
features table → Heuristic Detector → ml_results.heuristic_score

Advantages:
- No model training required (always available)
- Interpretable rules (explainable detection)
- Fast execution (no inference overhead)
- Detects known attack patterns reliably

Author: BGP Anomaly Detection System
Created: 2026-01-07
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('heuristic_detector.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class HeuristicDetector:
    """
    Rule-based BGP anomaly detector using domain knowledge.
    
    Detection Rules:
    1. High Withdrawal Rate: withdrawals/total_updates > 0.7
    2. Update Storm: total_updates > 1000 in 1-minute window
    3. Path Length Anomaly: avg path_length > 10 or < 2
    4. Peer Instability: unique_peers changes > 50%
    5. Announcement Spike: announcements > 500 in 1-minute
    """
    
    def __init__(self):
        """Initialize Heuristic Detector with database connection."""
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
        
        # Detection thresholds (configurable)
        self.thresholds = {
            'withdrawal_ratio_high': 0.7,
            'update_storm': 1000,
            'path_length_max': 10,
            'path_length_min': 2,
            'peer_change_percent': 0.5,
            'announcement_spike': 500
        }
        
        # Processing settings
        self.processing_interval = 10  # seconds
        self.batch_size = 100  # Process 100 features per batch
        
        # Statistics
        self.stats = {
            'features_processed': 0,
            'anomalies_detected': 0,
            'results_inserted': 0,
            'errors': 0,
            'last_processed_id': 0
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
    
    def get_last_processed_id(self) -> int:
        """
        Get the last processed feature ID from ml_results table.
        
        Returns:
            int: Last processed feature ID, or 0 if none
        """
        try:
            query = """
            SELECT MAX(id) 
            FROM features 
            WHERE id IN (SELECT feature_id FROM ml_results WHERE heuristic_score IS NOT NULL);
            """
            self.db_cursor.execute(query)
            result = self.db_cursor.fetchone()
            
            return result[0] if result and result[0] else 0
                
        except Exception as e:
            logger.error(f"Error getting last processed ID: {e}")
            return 0
    
    def detect_anomalies(self, feature: Dict) -> Tuple[float, List[str], bool]:
        """
        Apply heuristic rules to detect anomalies.
        
        Args:
            feature: Dictionary with feature values
            
        Returns:
            Tuple of (heuristic_score, triggered_rules, is_anomaly)
        """
        triggered_rules = []
        score = 0.0
        
        total_updates = feature.get('total_updates', 0)
        announcements = feature.get('announcements', 0)
        withdrawals = feature.get('withdrawals', 0)
        path_length = feature.get('path_length', 0.0)
        unique_peers = feature.get('unique_peers', 0)
        
        # Rule 1: High Withdrawal Rate
        if total_updates > 0:
            withdrawal_ratio = withdrawals / total_updates
            if withdrawal_ratio > self.thresholds['withdrawal_ratio_high']:
                triggered_rules.append(f"HIGH_WITHDRAWAL_RATE:{withdrawal_ratio:.2f}")
                score += 0.25
        
        # Rule 2: Update Storm
        if total_updates > self.thresholds['update_storm']:
            triggered_rules.append(f"UPDATE_STORM:{total_updates}")
            score += 0.20
        
        # Rule 3: Path Length Anomaly
        if path_length > self.thresholds['path_length_max']:
            triggered_rules.append(f"PATH_LENGTH_TOO_LONG:{path_length:.1f}")
            score += 0.15
        elif path_length > 0 and path_length < self.thresholds['path_length_min']:
            triggered_rules.append(f"PATH_LENGTH_TOO_SHORT:{path_length:.1f}")
            score += 0.15
        
        # Rule 4: Announcement Spike
        if announcements > self.thresholds['announcement_spike']:
            triggered_rules.append(f"ANNOUNCEMENT_SPIKE:{announcements}")
            score += 0.20
        
        # Rule 5: Peer Instability (if we have historical data)
        # This requires tracking changes over time - simplified for now
        if unique_peers == 0:
            triggered_rules.append("NO_PEERS")
            score += 0.10
        elif unique_peers > 100:  # Unusually high peer count
            triggered_rules.append(f"HIGH_PEER_COUNT:{unique_peers}")
            score += 0.10
        
        # Normalize score to [0, 1]
        score = min(score, 1.0)
        
        # Mark as anomaly if score > 0.5 or multiple rules triggered
        is_anomaly = score > 0.5 or len(triggered_rules) >= 2
        
        return score, triggered_rules, is_anomaly
    
    def process_features(self) -> int:
        """
        Process new features and generate heuristic detection results.
        
        Returns:
            int: Number of features processed
        """
        try:
            # Get last processed ID
            last_id = self.get_last_processed_id()
            
            # Fetch new features
            query = """
            SELECT id, timestamp, peer_addr, total_updates, announcements, 
                   withdrawals, path_length, unique_peers
            FROM features
            WHERE id > %s
            ORDER BY id ASC
            LIMIT %s;
            """
            
            self.db_cursor.execute(query, (last_id, self.batch_size))
            features = self.db_cursor.fetchall()
            
            if not features:
                logger.debug(f"No new features to process (last_id: {last_id})")
                return 0
            
            # Process each feature
            results = []
            for feature_row in features:
                feature_id, timestamp, peer_addr, total_updates, announcements, \
                    withdrawals, path_length, unique_peers = feature_row
                
                # Build feature dict
                feature_dict = {
                    'total_updates': total_updates or 0,
                    'announcements': announcements or 0,
                    'withdrawals': withdrawals or 0,
                    'path_length': path_length or 0.0,
                    'unique_peers': unique_peers or 0
                }
                
                # Run heuristic detection
                score, rules, is_anomaly = self.detect_anomalies(feature_dict)
                
                # Store result
                results.append({
                    'feature_id': feature_id,
                    'timestamp': timestamp,
                    'peer_addr': peer_addr,
                    'heuristic_score': score,
                    'heuristic_reasons': rules,
                    'heuristic_is_anomaly': is_anomaly
                })
                
                if is_anomaly:
                    self.stats['anomalies_detected'] += 1
            
            # Insert results into ml_results table
            inserted = self.insert_results(results)
            self.stats['features_processed'] += len(features)
            self.stats['results_inserted'] += inserted
            
            # Log progress every 50 features
            if self.stats['features_processed'] % 50 == 0:
                logger.info(f"[STATS] {self.stats['features_processed']} features | "
                          f"{self.stats['anomalies_detected']} anomalies | "
                          f"{self.stats['errors']} errors")
            
            return len(features)
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            self.stats['errors'] += 1
            return 0
    
    def insert_results(self, results: List[Dict]) -> int:
        """
        Insert heuristic detection results into ml_results table.
        
        Args:
            results: List of detection result dictionaries
            
        Returns:
            int: Number of records inserted
        """
        if not results:
            return 0
        
        try:
            query = """
            INSERT INTO ml_results (
                timestamp, peer_addr, feature_id,
                heuristic_score, heuristic_reasons, heuristic_is_anomaly,
                ensemble_score
            ) VALUES %s
            ON CONFLICT DO NOTHING;
            """
            
            # Prepare values (use heuristic_score as initial ensemble_score)
            values = [
                (
                    r['timestamp'], r['peer_addr'], r['feature_id'],
                    r['heuristic_score'], r['heuristic_reasons'], 
                    r['heuristic_is_anomaly'], r['heuristic_score']
                )
                for r in results
            ]
            
            execute_values(self.db_cursor, query, values)
            self.db_conn.commit()
            
            return len(results)
            
        except Exception as e:
            logger.error(f"Error inserting results: {e}")
            self.db_conn.rollback()
            self.stats['errors'] += 1
            return 0
    
    async def run_continuous(self):
        """
        Run heuristic detection continuously in a loop.
        """
        logger.info("=" * 60)
        logger.info("HEURISTIC DETECTOR SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"Processing interval: {self.processing_interval} seconds")
        logger.info(f"Batch size: {self.batch_size} features")
        logger.info("Detection rules: 5 active rules")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Connect to database
        if not self.connect_db():
            logger.error("Cannot start service without database connection")
            return
        
        try:
            while True:
                # Process pending features
                processed = self.process_features()
                
                if processed > 0:
                    logger.info(f"[OK] Processed {processed} features, "
                              f"detected {self.stats['anomalies_detected']} anomalies so far")
                
                # Wait before next cycle
                await asyncio.sleep(self.processing_interval)
                
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
            logger.info(f"Total features processed: {self.stats['features_processed']}")
            logger.info(f"Anomalies detected: {self.stats['anomalies_detected']}")
            logger.info(f"Results inserted: {self.stats['results_inserted']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("=" * 60)
    
    async def run_once(self):
        """
        Run heuristic detection once (for testing/batch processing).
        """
        logger.info("Running heuristic detection (single pass)...")
        
        if not self.connect_db():
            logger.error("Cannot run without database connection")
            return
        
        try:
            processed = self.process_features()
            logger.info(f"[OK] Processed {processed} features, detected {self.stats['anomalies_detected']} anomalies")
        finally:
            self.close_db()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BGP Heuristic Detector Service')
    parser.add_argument(
        '--mode',
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous (daemon) or once (batch)'
    )
    
    args = parser.parse_args()
    
    detector = HeuristicDetector()
    
    if args.mode == 'continuous':
        asyncio.run(detector.run_continuous())
    else:
        asyncio.run(detector.run_once())


if __name__ == "__main__":
    main()
