"""
LSTM Autoencoder Retraining Script
===================================
Retrains the LSTM Autoencoder model on fresh data when drift is detected.

Process:
1. Extract recent normal BGP data (last 7 days)
2. Create time-windowed sequences
3. Train new LSTM model
4. Validate reconstruction error
5. Hot-swap: Replace old model with new one
6. Continue detection without downtime
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import psycopg2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import json
import pickle
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class LSTMRetrainer:
    """Retrain LSTM Autoencoder model"""
    
    def __init__(self):
        # Database config
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Model paths with timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_dir = "model_output"
        self.history_dir = f"{self.model_dir}/history"
        os.makedirs(self.history_dir, exist_ok=True)
        
        # New timestamped model paths
        self.new_model_path = f"{self.history_dir}/lstm_model_{self.timestamp}.pkl"
        self.new_weights_path = f"{self.history_dir}/lstm_model_{self.timestamp}.weights.h5"
        self.new_metadata_path = f"{self.history_dir}/lstm_model_{self.timestamp}_metadata.json"
        self.new_scaler_path = f"{self.history_dir}/scaler_{self.timestamp}.pkl"
        
        # Production model paths (for hot-swap)
        self.prod_model_path = f"{self.model_dir}/lstm_model_for_pkl.weights.h5"
        self.prod_scaler_path = f"{self.model_dir}/scaler.pkl"
        self.prod_metadata_path = f"{self.model_dir}/lstm_metadata.json"
        
        # Training config
        self.training_days = 7
        self.time_steps = 10
        self.features_count = 9
        self.epochs = 50
        self.batch_size = 32
        self.training_start_time = None
        self.training_end_time = None
        
    def extract_training_data(self) -> pd.DataFrame:
        """Extract recent BGP features for training"""
        logger.info(f"Extracting training data from last {self.training_days} days...")
        
        conn = psycopg2.connect(**self.db_config)
        
        query = f"""
            SELECT 
                announcements, withdrawals, total_updates,
                withdrawal_ratio, flap_count, path_length,
                unique_peers, message_rate, session_resets,
                timestamp
            FROM features
            WHERE timestamp > NOW() - INTERVAL '{self.training_days} days'
            ORDER BY timestamp ASC
            LIMIT 50000;
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"✅ Extracted {len(df)} samples")
        return df
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create time-windowed sequences for LSTM"""
        logger.info(f"Creating sequences with {self.time_steps} time steps...")
        
        sequences = []
        for i in range(len(data) - self.time_steps):
            seq = data[i:i + self.time_steps]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        logger.info(f"✅ Created {len(sequences)} sequences")
        return sequences
    
    def build_model(self) -> Sequential:
        """Build LSTM Autoencoder architecture"""
        logger.info("Building LSTM Autoencoder...")
        
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.time_steps, self.features_count), return_sequences=False),
            RepeatVector(self.time_steps),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.features_count))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        logger.info("✅ Model architecture built")
        return model
    
    def train_model(self, model: Sequential, X: np.ndarray) -> Sequential:
        """Train LSTM model"""
        logger.info(f"Training LSTM model ({self.epochs} epochs)...")
        
        early_stop = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        logger.info(f"✅ Training complete (final loss: {history.history['loss'][-1]:.6f})")
        return model
    
    def validate_model(self, model: Sequential, X: np.ndarray) -> dict:
        """Validate model performance"""
        logger.info("Validating new model...")
        
        # Reconstruct data
        reconstructed = model.predict(X, verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean(np.abs(X - reconstructed), axis=(1, 2))
        
        metrics = {
            'mean_error': float(errors.mean()),
            'std_error': float(errors.std()),
            'error_range': [float(errors.min()), float(errors.max())],
            'threshold_95': float(np.percentile(errors, 95))
        }
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   Mean reconstruction error: {metrics['mean_error']:.6f}")
        logger.info(f"   95th percentile threshold: {metrics['threshold_95']:.6f}")
        
        return metrics
    
    def hot_swap_model(self):
        """Replace old model with new model (hot swap)"""
        logger.info("Performing hot swap...")
        
        # Backup current production model
        if os.path.exists(self.prod_model_path):
            import shutil
            backup_path = f"{self.model_dir}/lstm_model_backup_{self.timestamp}.weights.h5"
            shutil.copy2(self.prod_model_path, backup_path)
            logger.info(f"✅ Production model backed up: {backup_path}")
        
        # Copy new model to production
        import shutil
        shutil.copy2(self.new_weights_path, self.prod_model_path)
        logger.info(f"✅ New model activated: {self.prod_model_path}")
        
        # Copy new scaler to production
        if os.path.exists(self.new_scaler_path):
            shutil.copy2(self.new_scaler_path, self.prod_scaler_path)
            logger.info(f"✅ New scaler activated")
        
        # Copy metadata to production
        if os.path.exists(self.new_metadata_path):
            shutil.copy2(self.new_metadata_path, self.prod_metadata_path)
            logger.info(f"✅ Metadata updated")
        
        # Remove retraining flag if exists
        flag_path = f"model_artifacts/retrain_lstm.flag"
        if os.path.exists(flag_path):
            os.remove(flag_path)
            logger.info("✅ Retraining flag removed")
    
    def save_metadata(self, metrics: dict, training_info: dict):
        """Save comprehensive metadata JSON"""
        logger.info("Saving metadata...")
        
        # Get previous model info
        previous_model = None
        if os.path.exists(self.prod_metadata_path):
            try:
                with open(self.prod_metadata_path, 'r') as f:
                    prev_meta = json.load(f)
                    previous_model = prev_meta.get('model_file', 'unknown')
            except:
                pass
        
        # Read drift report if exists
        drift_info = {}
        drift_files = [f for f in os.listdir('model_artifacts') if f.startswith('drift_report_lstm_')]
        if drift_files:
            latest_drift = sorted(drift_files)[-1]
            try:
                with open(f"model_artifacts/{latest_drift}", 'r') as f:
                    drift_data = json.load(f)
                    drift_info = {
                        'score_shift': drift_data.get('metrics', {}).get('score_shift'),
                        'anomaly_rate_change': drift_data.get('metrics', {}).get('anomaly_rate_change'),
                        'reasons': drift_data.get('reasons', [])
                    }
            except:
                pass
        
        metadata = {
            'model_name': 'lstm_autoencoder',
            'model_file': f'lstm_model_{self.timestamp}.pkl',
            'weights_file': f'lstm_model_{self.timestamp}.weights.h5',
            'scaler_file': f'scaler_{self.timestamp}.pkl',
            'retrained_at': self.training_start_time.isoformat(),
            'training_completed_at': self.training_end_time.isoformat(),
            'training_duration_seconds': int((self.training_end_time - self.training_start_time).total_seconds()),
            'training_data': {
                'samples': training_info['samples'],
                'sequences': training_info['sequences'],
                'date_range_start': training_info['date_range'][0],
                'date_range_end': training_info['date_range'][1],
                'training_days': self.training_days
            },
            'architecture': {
                'type': 'LSTM_Autoencoder',
                'time_steps': self.time_steps,
                'features_count': self.features_count,
                'layers': 'LSTM(64) → RepeatVector → LSTM(64) → TimeDistributed(Dense(9))',
                'optimizer': 'adam',
                'loss': 'mse'
            },
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'validation_split': 0.2,
                'early_stopping_patience': 5
            },
            'performance_metrics': {
                'mean_reconstruction_error': metrics['mean_error'],
                'std_reconstruction_error': metrics['std_error'],
                'error_range_min': metrics['error_range'][0],
                'error_range_max': metrics['error_range'][1],
                'threshold_95_percentile': metrics['threshold_95']
            },
            'drift_trigger': drift_info if drift_info else {'triggered_by': 'manual'},
            'retrained_by': os.getenv('USER', 'system'),
            'previous_model': previous_model,
            'version': self.timestamp,
            'status': 'active'
        }
        
        # Save to timestamped file
        with open(self.new_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved: {self.new_metadata_path}")
        return metadata
    
    def run_retraining(self):
        """Execute complete retraining process"""
        logger.info("="*60)
        logger.info("LSTM AUTOENCODER RETRAINING")
        logger.info("="*60)
        
        self.training_start_time = datetime.now()
        
        try:
            # 1. Extract data
            df = self.extract_training_data()
            
            if len(df) < 1000:
                logger.error("❌ Insufficient data for retraining (need at least 1000 samples)")
                return False
            
            # Store date range for metadata
            date_range = (df['timestamp'].min().isoformat(), df['timestamp'].max().isoformat())
            
            # 2. Prepare features
            feature_cols = ['announcements', 'withdrawals', 'total_updates',
                          'withdrawal_ratio', 'flap_count', 'path_length',
                          'unique_peers', 'message_rate', 'session_resets']
            
            data = df[feature_cols].values
            
            # Normalize
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # 3. Create sequences
            X = self.create_sequences(data_scaled)
            
            if len(X) < 100:
                logger.error("❌ Insufficient sequences for training")
                return False
            
            # 4. Build and train model
            model = self.build_model()
            model = self.train_model(model, X)
            
            # 5. Validate model
            metrics = self.validate_model(model, X)
            
            self.training_end_time = datetime.now()
            
            # 6. Save new model with timestamp
            model.save_weights(self.new_weights_path)
            logger.info(f"✅ New model saved: {self.new_weights_path}")
            
            # Save scaler with timestamp
            with open(self.new_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"✅ Scaler saved: {self.new_scaler_path}")
            
            # 7. Save comprehensive metadata
            training_info = {
                'samples': len(df),
                'sequences': len(X),
                'date_range': date_range
            }
            metadata = self.save_metadata(metrics, training_info)
            
            # 8. Hot swap
            self.hot_swap_model()
            
            # 9. Update database drift status
            self.update_drift_status_db()
            
            logger.info("="*60)
            logger.info("✅ RETRAINING COMPLETE")
            logger.info("="*60)
            logger.info(f"Model file: {os.path.basename(self.new_model_path)}")
            logger.info(f"Metadata: {os.path.basename(self.new_metadata_path)}")
            logger.info(f"Training duration: {metadata['training_duration_seconds']}s")
            logger.info("LSTM detector will automatically load new model")
            logger.info("on next detection cycle (within 10 seconds)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_drift_status_db(self):
        """Update drift status in database after successful retraining"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE drift_status 
                SET drift_detected = FALSE,
                    status = 'retrained',
                    message = 'Model successfully retrained',
                    retraining_completed_at = %s
                WHERE model_name = 'lstm';
            """, (self.training_end_time,))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ Database drift status updated")
        except Exception as e:
            logger.warning(f"⚠️ Could not update database: {e}")


if __name__ == "__main__":
    retrainer = LSTMRetrainer()
    success = retrainer.run_retraining()
    sys.exit(0 if success else 1)
