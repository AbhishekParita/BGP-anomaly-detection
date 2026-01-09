"""
Ensemble Coordinator Service

This service combines detection results from multiple ML models to generate
a final ensemble score with improved accuracy and robustness.

Architecture Flow:
ml_results (heuristic_score, lstm_anomaly_score, if_anomaly_score) 
    → Ensemble Coordinator → ml_results.ensemble_score + ensemble_is_anomaly

Key Features:
- Weighted voting across multiple detectors
- Graceful degradation (works with 1-3 models)
- Configurable weights for each detector
- Handles missing model scores

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
        logging.FileHandler('ensemble_coordinator.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EnsembleCoordinator:
    """
    Ensemble coordinator that combines multiple ML detector outputs.
    
    Ensemble Strategy:
    - Weighted average of available detector scores
    - Configurable weights for each detector
    - Graceful degradation: if 1-2 models missing, use available ones
    - Final decision: ensemble_score > threshold
    """
    
    def __init__(self):
        """Initialize Ensemble Coordinator with database connection."""
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
        
        # Model weights (configurable)
        # Higher weight = more trust in that model's predictions
        self.weights = {
            'heuristic': 0.25,      # Rule-based, explainable, fast
            'lstm': 0.40,           # Temporal patterns, low false positives
            'isolation_forest': 0.35 # Statistical outliers, high sensitivity
        }
        
        # Ensemble decision threshold
        self.anomaly_threshold = 0.4  # Lowered from 0.5 to catch more anomalies
        
        # Processing settings
        self.processing_interval = 10  # seconds
        self.batch_size = 100  # Process 100 records per batch
        
        # Statistics
        self.stats = {
            'records_processed': 0,
            'ensemble_anomalies': 0,
            'models_available': {'heuristic': 0, 'lstm': 0, 'if': 0},
            'records_updated': 0,
            'errors': 0
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
    
    def calculate_ensemble_score(self, scores: Dict) -> Tuple[float, bool, Dict]:
        """
        Calculate weighted ensemble score from available detector scores.
        
        Args:
            scores: Dictionary with detector scores
                {
                    'heuristic_score': float or None,
                    'lstm_anomaly_score': float or None,
                    'if_anomaly_score': float or None
                }
        
        Returns:
            Tuple of (ensemble_score, is_anomaly, metadata)
        """
        available_models = []
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Heuristic model
        if scores.get('heuristic_score') is not None:
            available_models.append('heuristic')
            weighted_sum += scores['heuristic_score'] * self.weights['heuristic']
            total_weight += self.weights['heuristic']
        
        # LSTM model
        if scores.get('lstm_anomaly_score') is not None:
            available_models.append('lstm')
            weighted_sum += scores['lstm_anomaly_score'] * self.weights['lstm']
            total_weight += self.weights['lstm']
        
        # Isolation Forest model
        if scores.get('if_anomaly_score') is not None:
            available_models.append('isolation_forest')
            weighted_sum += scores['if_anomaly_score'] * self.weights['isolation_forest']
            total_weight += self.weights['isolation_forest']
        
        # Calculate ensemble score
        if total_weight > 0:
            ensemble_score = weighted_sum / total_weight
        else:
            # No models available - default to 0 (no anomaly)
            ensemble_score = 0.0
        
        # Determine anomaly status
        is_anomaly = ensemble_score > self.anomaly_threshold
        
        # Metadata
        metadata = {
            'models_used': available_models,
            'model_count': len(available_models),
            'ensemble_score': ensemble_score
        }
        
        return ensemble_score, is_anomaly, metadata
    
    def get_pending_records(self) -> List[Dict]:
        """
        Fetch ml_results records that need ensemble score calculation.
        
        Returns records where:
        - At least one detector score is present
        - Multiple detector scores available (for true ensemble)
        
        Returns:
            List of record dictionaries
        """
        try:
            query = """
            SELECT id, feature_id, timestamp, peer_addr,
                   heuristic_score, heuristic_is_anomaly,
                   lstm_anomaly_score, lstm_is_anomaly,
                   if_anomaly_score, if_is_anomaly,
                   ensemble_score, ensemble_is_anomaly
            FROM ml_results
            WHERE (
                heuristic_score IS NOT NULL 
                OR lstm_anomaly_score IS NOT NULL 
                OR if_anomaly_score IS NOT NULL
            )
            AND ensemble_score IS NULL
            ORDER BY id ASC
            LIMIT %s;
            """
            
            self.db_cursor.execute(query, (self.batch_size,))
            rows = self.db_cursor.fetchall()
            
            records = []
            for row in rows:
                records.append({
                    'id': row[0],
                    'feature_id': row[1],
                    'timestamp': row[2],
                    'peer_addr': row[3],
                    'heuristic_score': row[4],
                    'heuristic_is_anomaly': row[5],
                    'lstm_anomaly_score': row[6],
                    'lstm_is_anomaly': row[7],
                    'if_anomaly_score': row[8],
                    'if_is_anomaly': row[9],
                    'ensemble_score': row[10],
                    'ensemble_is_anomaly': row[11]
                })
            
            return records
            
        except Exception as e:
            logger.error(f"Error fetching pending records: {e}")
            return []
    
    def process_records(self) -> int:
        """
        Process pending ml_results records and calculate ensemble scores.
        
        Returns:
            int: Number of records processed
        """
        try:
            # Fetch pending records
            records = self.get_pending_records()
            
            if not records:
                logger.debug("No pending records to process")
                return 0
            
            # Process each record
            updates = []
            for record in records:
                # Extract detector scores
                scores = {
                    'heuristic_score': record['heuristic_score'],
                    'lstm_anomaly_score': record['lstm_anomaly_score'],
                    'if_anomaly_score': record['if_anomaly_score']
                }
                
                # Calculate ensemble score
                ensemble_score, is_anomaly, metadata = self.calculate_ensemble_score(scores)
                
                # Track model availability
                for model in metadata['models_used']:
                    if model in self.stats['models_available']:
                        self.stats['models_available'][model] += 1
                
                # Prepare update
                updates.append({
                    'id': record['id'],
                    'ensemble_score': ensemble_score,
                    'ensemble_is_anomaly': is_anomaly
                })
                
                if is_anomaly:
                    self.stats['ensemble_anomalies'] += 1
            
            # Update database
            updated = self.update_ensemble_scores(updates)
            self.stats['records_processed'] += len(records)
            self.stats['records_updated'] += updated
            
            # Debug: Log anomaly updates
            anomaly_count = sum(1 for u in updates if u['ensemble_is_anomaly'])
            if anomaly_count > 0:
                logger.info(f"[DEBUG] Updating {anomaly_count} records as anomalies out of {len(updates)} total")
            
            # Log progress every 50 records
            if self.stats['records_processed'] % 50 == 0:
                logger.info(f"[STATS] {self.stats['records_processed']} records | "
                          f"{self.stats['ensemble_anomalies']} anomalies | "
                          f"Models: H={self.stats['models_available']['heuristic']} "
                          f"L={self.stats['models_available']['lstm']} "
                          f"IF={self.stats['models_available']['if']}")
            
            return len(records)
            
        except Exception as e:
            logger.error(f"Error processing records: {e}")
            self.stats['errors'] += 1
            return 0
    
    def update_ensemble_scores(self, updates: List[Dict]) -> int:
        """
        Update ensemble scores in ml_results table.
        
        Args:
            updates: List of update dictionaries with id, ensemble_score, ensemble_is_anomaly
        
        Returns:
            int: Number of records updated
        """
        if not updates:
            return 0
        
        try:
            query = """
            UPDATE ml_results
            SET ensemble_score = %s,
                ensemble_is_anomaly = %s
            WHERE id = %s;
            """
            
            values = [
                (u['ensemble_score'], u['ensemble_is_anomaly'], u['id'])
                for u in updates
            ]
            
            # Debug: Log a sample update
            if values:
                logger.debug(f"Sample update: score={values[0][0]}, is_anomaly={values[0][1]}, id={values[0][2]}")
            
            self.db_cursor.executemany(query, values)
            rows_affected = self.db_cursor.rowcount
            self.db_conn.commit()
            
            logger.debug(f"UPDATE affected {rows_affected} rows")
            
            return len(updates)
            
        except Exception as e:
            logger.error(f"Error updating ensemble scores: {e}")
            self.db_conn.rollback()
            self.stats['errors'] += 1
            return 0
    
    async def run_continuous(self):
        """
        Run ensemble coordination continuously in a loop.
        """
        logger.info("=" * 60)
        logger.info("ENSEMBLE COORDINATOR SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"Model weights: Heuristic={self.weights['heuristic']:.2f}, "
                   f"LSTM={self.weights['lstm']:.2f}, IF={self.weights['isolation_forest']:.2f}")
        logger.info(f"Anomaly threshold: {self.anomaly_threshold}")
        logger.info(f"Processing interval: {self.processing_interval} seconds")
        logger.info(f"Batch size: {self.batch_size} records")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Connect to database
        if not self.connect_db():
            logger.error("Cannot start service without database connection")
            return
        
        try:
            while True:
                # Process pending records
                processed = self.process_records()
                
                if processed > 0:
                    logger.info(f"[OK] Processed {processed} records, "
                              f"detected {self.stats['ensemble_anomalies']} anomalies so far")
                
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
            logger.info(f"Total records processed: {self.stats['records_processed']}")
            logger.info(f"Ensemble anomalies detected: {self.stats['ensemble_anomalies']}")
            logger.info(f"Records updated: {self.stats['records_updated']}")
            logger.info(f"Model availability:")
            logger.info(f"  - Heuristic: {self.stats['models_available']['heuristic']}")
            logger.info(f"  - LSTM: {self.stats['models_available']['lstm']}")
            logger.info(f"  - Isolation Forest: {self.stats['models_available']['if']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("=" * 60)
    
    async def run_once(self):
        """
        Run ensemble coordination once (for testing/batch processing).
        """
        logger.info("Running ensemble coordination (single pass)...")
        logger.info(f"Weights: H={self.weights['heuristic']:.2f}, "
                   f"L={self.weights['lstm']:.2f}, IF={self.weights['isolation_forest']:.2f}")
        
        if not self.connect_db():
            logger.error("Cannot run without database connection")
            return
        
        try:
            processed = self.process_records()
            logger.info(f"[OK] Processed {processed} records, "
                      f"detected {self.stats['ensemble_anomalies']} anomalies")
            logger.info(f"Model usage: H={self.stats['models_available']['heuristic']}, "
                      f"L={self.stats['models_available']['lstm']}, "
                      f"IF={self.stats['models_available']['if']}")
        finally:
            self.close_db()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BGP Ensemble Coordinator Service')
    parser.add_argument(
        '--mode',
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous (daemon) or once (batch)'
    )
    
    args = parser.parse_args()
    
    coordinator = EnsembleCoordinator()
    
    if args.mode == 'continuous':
        asyncio.run(coordinator.run_continuous())
    else:
        asyncio.run(coordinator.run_once())


if __name__ == "__main__":
    main()
