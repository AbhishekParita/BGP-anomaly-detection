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
        
        # Model paths
        self.model_dir = "model_output"
        self.old_model_path = f"{self.model_dir}/lstm_model_for_pkl.weights.h5"
        self.new_model_path = f"{self.model_dir}/lstm_model_for_pkl_new.weights.h5"
        self.backup_path = f"{self.model_dir}/lstm_model_for_pkl_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.weights.h5"
        
        # Training config
        self.training_days = 7
        self.time_steps = 10
        self.features_count = 9
        self.epochs = 50
        self.batch_size = 32
        
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
        
        # Backup old model
        if os.path.exists(self.old_model_path):
            import shutil
            shutil.copy2(self.old_model_path, self.backup_path)
            logger.info(f"✅ Old model backed up: {self.backup_path}")
        
        # Replace with new model
        import shutil
        shutil.move(self.new_model_path, self.old_model_path)
        logger.info(f"✅ New model activated: {self.old_model_path}")
        
        # Remove retraining flag
        flag_path = f"model_artifacts/retrain_lstm.flag"
        if os.path.exists(flag_path):
            os.remove(flag_path)
            logger.info("✅ Retraining flag removed")
    
    def run_retraining(self):
        """Execute complete retraining process"""
        logger.info("="*60)
        logger.info("LSTM AUTOENCODER RETRAINING")
        logger.info("="*60)
        
        try:
            # 1. Extract data
            df = self.extract_training_data()
            
            if len(df) < 1000:
                logger.error("❌ Insufficient data for retraining (need at least 1000 samples)")
                return False
            
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
            
            # 6. Save new model
            model.save_weights(self.new_model_path)
            logger.info(f"✅ New model saved: {self.new_model_path}")
            
            # Save scaler
            scaler_path = f"{self.model_dir}/scaler_new.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 7. Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(df),
                'sequences': len(X),
                'training_days': self.training_days,
                'time_steps': self.time_steps,
                'epochs': self.epochs,
                'validation_metrics': metrics
            }
            
            with open(f"{self.model_dir}/lstm_metadata_new.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 8. Hot swap
            self.hot_swap_model()
            
            # Also update scaler
            old_scaler_path = f"{self.model_dir}/scaler.pkl"
            if os.path.exists(old_scaler_path):
                import shutil
                shutil.copy2(old_scaler_path, old_scaler_path.replace('.pkl', '_backup.pkl'))
            import shutil
            shutil.move(scaler_path, old_scaler_path)
            
            logger.info("="*60)
            logger.info("✅ RETRAINING COMPLETE")
            logger.info("="*60)
            logger.info("LSTM detector will automatically load new model")
            logger.info("on next detection cycle (within 10 seconds)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    retrainer = LSTMRetrainer()
    success = retrainer.run_retraining()
    sys.exit(0 if success else 1)
