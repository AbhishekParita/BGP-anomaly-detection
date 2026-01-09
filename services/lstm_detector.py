"""
LSTM Detector Service

This service runs LSTM Autoencoder anomaly detection on aggregated BGP features.
It loads the pre-trained model and performs inference independently.

Architecture Flow:
features table → LSTM Detector → ml_results.lstm_anomaly_score

Model Details:
- Type: LSTM Autoencoder (sequence-to-sequence)
- Input: 10 timesteps x 9 features
- Output: Reconstruction error (0-1 normalized)
- Threshold: 0.37 (from training)

Advantages:
- Detects temporal patterns and anomalies
- Learns normal behavior from historical data
- Good for detecting unknown attack patterns
- Can be retrained independently without affecting other models

Author: BGP Anomaly Detection System
Created: 2026-01-07
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
except ImportError as e:
    print(f"Error: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lstm_detector.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LSTMDetector:
    """
    LSTM Autoencoder-based BGP anomaly detector.
    
    The model learns normal BGP behavior patterns and identifies anomalies
    based on reconstruction error. High reconstruction error = anomaly.
    """
    
    def __init__(self, model_path: str = "model_output/lstm/lstm_best.h5",
                 config_path: str = "model_output/config.json"):
        """Initialize LSTM Detector."""
        self.db_conn = None
        self.db_cursor = None
        
        # Model paths
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        
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
        self.seq_len = 10  # Sequence length for LSTM
        
        # Statistics
        self.stats = {
            'features_processed': 0,
            'anomalies_detected': 0,
            'results_inserted': 0,
            'errors': 0,
            'last_processed_id': 0,
            'model_version': 'unknown'
        }
    
    def load_model(self) -> bool:
        """
        Load pre-trained LSTM model and configuration.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load configuration
            if not os.path.exists(self.config_path):
                logger.error(f"Config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.seq_len = self.config.get('seq_len', 10)
            self.threshold = self.config.get('threshold', 0.37)
            self.feature_names = self.config.get('feature_names', [])
            
            logger.info(f"[OK] Loaded config: seq_len={self.seq_len}, threshold={self.threshold}")
            
            # Load LSTM model
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load with compile=False to avoid metric compatibility issues
            self.model = keras.models.load_model(self.model_path, compile=False)
            self.stats['model_version'] = self.config.get('saved_at', 'unknown')
            
            logger.info(f"[OK] Loaded LSTM model from {self.model_path}")
            logger.info(f"Model version: {self.stats['model_version']}")
            
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
            WHERE id IN (SELECT feature_id FROM ml_results WHERE lstm_anomaly_score IS NOT NULL);
            """
            self.db_cursor.execute(query)
            result = self.db_cursor.fetchone()
            
            return result[0] if result and result[0] else 0
                
        except Exception as e:
            logger.error(f"Error getting last processed ID: {e}")
            return 0
    
    def fetch_sequence_data(self, feature_id: int) -> Optional[np.ndarray]:
        """
        Fetch sequence of features (10 timesteps) for LSTM input.
        
        Args:
            feature_id: ID of the current feature
            
        Returns:
            numpy array of shape (10, 9) or None if insufficient data
        """
        try:
            # Fetch last 10 features (including current)
            query = """
            SELECT announcements, withdrawals, total_updates, 
                   CASE WHEN total_updates > 0 
                        THEN CAST(withdrawals AS FLOAT) / total_updates 
                        ELSE 0.0 END as withdrawal_ratio,
                   0 as flap_count,
                   path_length, unique_peers, 
                   CASE WHEN total_updates > 0 
                        THEN CAST(total_updates AS FLOAT) / 60.0 
                        ELSE 0.0 END as message_rate,
                   0 as session_resets
            FROM features
            WHERE id <= %s
            ORDER BY id DESC
            LIMIT %s;
            """
            
            self.db_cursor.execute(query, (feature_id, self.seq_len))
            rows = self.db_cursor.fetchall()
            
            if len(rows) < self.seq_len:
                logger.debug(f"Insufficient history for feature {feature_id}: {len(rows)}/{self.seq_len}")
                return None
            
            # Convert to numpy array and reverse (oldest to newest)
            sequence = np.array(rows[::-1], dtype=np.float32)
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error fetching sequence data: {e}")
            return None
    
    def predict_anomaly(self, sequence: np.ndarray) -> tuple:
        """
        Run LSTM inference and calculate anomaly score.
        
        Args:
            sequence: Input sequence (10, 9)
            
        Returns:
            tuple: (reconstruction_error, anomaly_score, is_anomaly)
        """
        try:
            # Reshape for model input: (1, seq_len, features)
            X = sequence.reshape(1, self.seq_len, -1)
            
            # Normalize (simple min-max, ideally should use training scaler)
            # For now, use reasonable ranges based on BGP data
            ranges = np.array([1000, 500, 1500, 1.0, 100, 15, 200, 30, 50], dtype=np.float32)
            X_norm = X / ranges
            X_norm = np.clip(X_norm, 0, 1)
            
            # Predict (reconstruct)
            X_reconstructed = self.model.predict(X_norm, verbose=0)
            
            # Calculate reconstruction error (MSE)
            mse = np.mean(np.square(X_norm - X_reconstructed))
            reconstruction_error = float(mse)
            
            # Normalize error to [0, 1] score
            # Using sigmoid to map reconstruction error to probability
            anomaly_score = 1.0 / (1.0 + np.exp(-10 * (reconstruction_error - 0.1)))
            anomaly_score = float(np.clip(anomaly_score, 0, 1))
            
            # Binary classification
            # Use anomaly_score (0-1 normalized) for threshold comparison
            # Threshold of 0.5 means 50% confidence that it's an anomaly
            is_anomaly = anomaly_score > 0.5
            
            return reconstruction_error, anomaly_score, is_anomaly
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return 0.0, 0.0, False
    
    def process_features(self) -> int:
        """
        Process new features and generate LSTM detection results.
        
        Returns:
            int: Number of features processed
        """
        try:
            # Get last processed ID
            last_id = self.get_last_processed_id()
            
            # Fetch new features (just IDs, we'll get sequences individually)
            query = """
            SELECT id, timestamp, peer_addr
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
            for feature_id, timestamp, peer_addr in features:
                # Fetch sequence for LSTM
                sequence = self.fetch_sequence_data(feature_id)
                
                if sequence is None:
                    # Skip if insufficient history
                    continue
                
                # Run LSTM prediction
                recon_error, anomaly_score, is_anomaly = self.predict_anomaly(sequence)
                
                # Store result
                results.append({
                    'feature_id': feature_id,
                    'timestamp': timestamp,
                    'peer_addr': peer_addr,
                    'lstm_reconstruction_error': recon_error,
                    'lstm_anomaly_score': anomaly_score,
                    'lstm_is_anomaly': is_anomaly
                })
                
                if is_anomaly:
                    self.stats['anomalies_detected'] += 1
            
            # Insert results into ml_results table
            if results:
                inserted = self.insert_results(results)
                self.stats['features_processed'] += len(results)
                self.stats['results_inserted'] += inserted
            
            # Log progress every 50 features
            if self.stats['features_processed'] % 50 == 0:
                logger.info(f"[STATS] {self.stats['features_processed']} features | "
                          f"{self.stats['anomalies_detected']} anomalies | "
                          f"{self.stats['errors']} errors")
            
            return len(results)
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            self.stats['errors'] += 1
            return 0
    
    def insert_results(self, results: List[Dict]) -> int:
        """
        Insert LSTM detection results into ml_results table.
        
        Args:
            results: List of detection result dictionaries
            
        Returns:
            int: Number of records inserted
        """
        if not results:
            return 0
        
        try:
            # First, check which features already have LSTM results
            existing_query = """
            SELECT feature_id FROM ml_results 
            WHERE feature_id = ANY(%s) AND lstm_anomaly_score IS NOT NULL;
            """
            feature_ids = [r['feature_id'] for r in results]
            self.db_cursor.execute(existing_query, (feature_ids,))
            existing_ids = {row[0] for row in self.db_cursor.fetchall()}
            
            # Filter to only new results
            new_results = [r for r in results if r['feature_id'] not in existing_ids]
            
            if not new_results:
                return 0
            
            query = """
            INSERT INTO ml_results (
                timestamp, peer_addr, feature_id,
                lstm_reconstruction_error, lstm_anomaly_score, lstm_is_anomaly,
                model_version
            ) VALUES %s;
            """
            
            # Prepare values - let Ensemble Coordinator calculate ensemble_score
            values = [
                (
                    r['timestamp'], r['peer_addr'], r['feature_id'],
                    r['lstm_reconstruction_error'], r['lstm_anomaly_score'],
                    r['lstm_is_anomaly'],
                    self.stats['model_version']
                )
                for r in new_results
            ]
            
            execute_values(self.db_cursor, query, values)
            self.db_conn.commit()
            
            return len(new_results)
            
        except Exception as e:
            logger.error(f"Error inserting results: {e}")
            self.db_conn.rollback()
            self.stats['errors'] += 1
            return 0
    
    async def run_continuous(self):
        """
        Run LSTM detection continuously in a loop.
        """
        logger.info("=" * 60)
        logger.info("LSTM DETECTOR SERVICE STARTED")
        logger.info("=" * 60)
        
        # Load model first
        if not self.load_model():
            logger.error("Failed to load LSTM model, cannot start service")
            return
        
        logger.info(f"Processing interval: {self.processing_interval} seconds")
        logger.info(f"Batch size: {self.batch_size} features")
        logger.info(f"Sequence length: {self.seq_len} timesteps")
        logger.info(f"Anomaly threshold: {self.threshold}")
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
            logger.info(f"Model version: {self.stats['model_version']}")
            logger.info("=" * 60)
    
    async def run_once(self):
        """
        Run LSTM detection once (for testing/batch processing).
        """
        logger.info("Running LSTM detection (single pass)...")
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load LSTM model")
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
    
    parser = argparse.ArgumentParser(description='BGP LSTM Detector Service')
    parser.add_argument(
        '--mode',
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous (daemon) or once (batch)'
    )
    parser.add_argument(
        '--model',
        default='model_output/lstm/lstm_best.h5',
        help='Path to LSTM model file'
    )
    parser.add_argument(
        '--config',
        default='model_output/config.json',
        help='Path to model config file'
    )
    
    args = parser.parse_args()
    
    detector = LSTMDetector(model_path=args.model, config_path=args.config)
    
    if args.mode == 'continuous':
        asyncio.run(detector.run_continuous())
    else:
        asyncio.run(detector.run_once())


if __name__ == "__main__":
    main()
