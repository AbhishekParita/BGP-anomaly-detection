"""
Isolation Forest Detector Service

This service runs Isolation Forest anomaly detection on aggregated BGP features.
It uses a pre-trained sklearn IsolationForest model for unsupervised anomaly detection.

Architecture Flow:
features table → Isolation Forest Detector → ml_results.if_anomaly_score

Advantages:
- Unsupervised learning (no labeled data needed)
- Handles high-dimensional features well
- Fast inference time
- Detects global anomalies effectively

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
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('isolation_forest_detector.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class IsolationForestDetector:
    """
    Isolation Forest-based BGP anomaly detector using pre-trained sklearn model.
    
    Detection Method:
    - Uses IsolationForest to detect outliers in feature space
    - Anomaly score: -1 = anomaly, 1 = normal
    - Decision function: negative = anomaly, positive = normal
    """
    
    def __init__(self, model_path: str = 'model_artifacts/iso_forest_bgp_production.pkl'):
        """
        Initialize Isolation Forest Detector.
        
        Args:
            model_path: Path to pre-trained IsolationForest model (.pkl)
        """
        self.db_conn = None
        self.db_cursor = None
        self.model = None
        self.model_path = model_path
        self.model_loaded = False
        
        # Database configuration
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Processing settings
        self.processing_interval = 10  # seconds
        self.batch_size = 100  # Process 100 features per batch
        
        # Feature names expected by the model
        self.feature_names = [
            'total_updates', 'announcements', 'withdrawals',
            'path_length', 'unique_peers', 'unique_prefixes',
            'avg_prefix_len', 'unique_origins', 'unique_next_hops'
        ]
        
        # Statistics
        self.stats = {
            'features_processed': 0,
            'anomalies_detected': 0,
            'results_inserted': 0,
            'errors': 0
        }
    
    def load_model(self) -> bool:
        """
        Load pre-trained Isolation Forest model from disk.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"[ERROR] Model file not found: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            
            logger.info(f"[OK] Loaded Isolation Forest model from {self.model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Log model parameters
            if hasattr(self.model, 'n_estimators'):
                logger.info(f"n_estimators: {self.model.n_estimators}")
            if hasattr(self.model, 'contamination'):
                logger.info(f"contamination: {self.model.contamination}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            return False
    
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
            WHERE id IN (SELECT feature_id FROM ml_results WHERE if_anomaly_score IS NOT NULL);
            """
            self.db_cursor.execute(query)
            result = self.db_cursor.fetchone()
            
            return result[0] if result and result[0] else 0
                
        except Exception as e:
            logger.error(f"Error getting last processed ID: {e}")
            return 0
    
    def prepare_features(self, feature_row: tuple) -> Optional[np.ndarray]:
        """
        Prepare feature vector for Isolation Forest inference.
        
        Args:
            feature_row: Database row with feature values (only 8 columns available)
            
        Returns:
            np.ndarray: Feature vector shaped (1, n_features), or None if error
        """
        try:
            # Unpack available feature columns from database
            feature_id, timestamp, peer_addr, total_updates, announcements, \
                withdrawals, path_length, unique_peers = feature_row
            
            # Build feature vector with available features + zeros for missing ones
            # Model expects 9 features, but database only has 5 statistical features
            # Pad remaining features with zeros (model trained on full feature set)
            feature_vector = np.array([
                total_updates or 0,
                announcements or 0,
                withdrawals or 0,
                path_length or 0.0,
                unique_peers or 0,
                0,  # unique_prefixes (not in database)
                0.0,  # avg_prefix_len (not in database)
                0,  # unique_origins (not in database)
                0   # unique_next_hops (not in database)
            ], dtype=np.float32)
            
            # Reshape to (1, n_features) for prediction
            return feature_vector.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def predict_anomaly(self, feature_vector: np.ndarray) -> tuple:
        """
        Run Isolation Forest inference to detect anomalies.
        
        Args:
            feature_vector: Numpy array of shape (1, n_features)
            
        Returns:
            tuple: (anomaly_score, is_anomaly, decision_score)
                - anomaly_score: 0.0-1.0 (higher = more anomalous)
                - is_anomaly: Boolean
                - decision_score: Raw decision function output
        """
        try:
            # Get prediction: -1 = anomaly, 1 = normal
            prediction = self.model.predict(feature_vector)[0]
            
            # Get anomaly score (decision function)
            # Negative = anomalous, Positive = normal
            decision_score = self.model.decision_function(feature_vector)[0]
            
            # Convert to normalized score [0, 1] where higher = more anomalous
            # Use sigmoid transformation: score = 1 / (1 + e^decision)
            anomaly_score = 1.0 / (1.0 + np.exp(decision_score))
            
            # Determine if anomaly
            is_anomaly = (prediction == -1)
            
            return float(anomaly_score), is_anomaly, float(decision_score)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0.5, False, 0.0  # Default values
    
    def process_features(self) -> int:
        """
        Process new features and generate Isolation Forest detection results.
        
        Returns:
            int: Number of features processed
        """
        if not self.model_loaded:
            logger.error("Model not loaded, cannot process features")
            return 0
        
        try:
            # Get last processed ID
            last_id = self.get_last_processed_id()
            
            # Fetch new features (only available columns)
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
                feature_id = feature_row[0]
                timestamp = feature_row[1]
                peer_addr = feature_row[2]
                
                # Prepare feature vector
                feature_vector = self.prepare_features(feature_row)
                if feature_vector is None:
                    continue
                
                # Run Isolation Forest inference
                anomaly_score, is_anomaly, decision_score = self.predict_anomaly(feature_vector)
                
                # Store result
                results.append({
                    'feature_id': feature_id,
                    'timestamp': timestamp,
                    'peer_addr': peer_addr,
                    'if_anomaly_score': anomaly_score,
                    'if_is_anomaly': is_anomaly
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
        Insert Isolation Forest detection results into ml_results table.
        
        Args:
            results: List of detection result dictionaries
            
        Returns:
            int: Number of records inserted
        """
        if not results:
            return 0
        
        try:
            # First, check which feature_ids already exist
            feature_ids = [r['feature_id'] for r in results]
            
            check_query = """
            SELECT feature_id FROM ml_results WHERE feature_id = ANY(%s);
            """
            self.db_cursor.execute(check_query, (feature_ids,))
            existing_ids = set(row[0] for row in self.db_cursor.fetchall())
            
            # For existing records, UPDATE
            update_results = [r for r in results if r['feature_id'] in existing_ids]
            if update_results:
                update_query = """
                UPDATE ml_results
                SET if_anomaly_score = %s,
                    if_is_anomaly = %s
                WHERE feature_id = %s;
                """
                update_values = [
                    (float(r['if_anomaly_score']), 
                     bool(r['if_is_anomaly']), r['feature_id'])
                    for r in update_results
                ]
                self.db_cursor.executemany(update_query, update_values)
            
            # For new records, INSERT
            insert_results = [r for r in results if r['feature_id'] not in existing_ids]
            if insert_results:
                insert_query = """
                INSERT INTO ml_results (
                    timestamp, peer_addr, feature_id,
                    if_anomaly_score, if_is_anomaly
                ) VALUES %s;
                """
                # Let Ensemble Coordinator calculate ensemble_score
                insert_values = [
                    (r['timestamp'], r['peer_addr'], r['feature_id'],
                     float(r['if_anomaly_score']), 
                     bool(r['if_is_anomaly']))
                    for r in insert_results
                ]
                execute_values(self.db_cursor, insert_query, insert_values)
            
            self.db_conn.commit()
            return len(results)
            
        except Exception as e:
            logger.error(f"Error inserting results: {e}")
            self.db_conn.rollback()
            self.stats['errors'] += 1
            return 0
    
    async def run_continuous(self):
        """
        Run Isolation Forest detection continuously in a loop.
        """
        logger.info("=" * 60)
        logger.info("ISOLATION FOREST DETECTOR SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Processing interval: {self.processing_interval} seconds")
        logger.info(f"Batch size: {self.batch_size} features")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Load model
        if not self.load_model():
            logger.error("Cannot start service without model")
            return
        
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
        Run Isolation Forest detection once (for testing/batch processing).
        """
        logger.info("Running Isolation Forest detection (single pass)...")
        
        # Load model
        if not self.load_model():
            logger.error("Cannot run without model")
            return
        
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
    
    parser = argparse.ArgumentParser(description='BGP Isolation Forest Detector Service')
    parser.add_argument(
        '--mode',
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous (daemon) or once (batch)'
    )
    parser.add_argument(
        '--model-path',
        default='model_artifacts/iso_forest_bgp_production.pkl',
        help='Path to pre-trained Isolation Forest model (.pkl)'
    )
    
    args = parser.parse_args()
    
    detector = IsolationForestDetector(model_path=args.model_path)
    
    if args.mode == 'continuous':
        asyncio.run(detector.run_continuous())
    else:
        asyncio.run(detector.run_once())


if __name__ == "__main__":
    main()
