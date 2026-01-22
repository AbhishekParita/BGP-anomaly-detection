"""
Isolation Forest Retraining Script
===================================
Retrains the Isolation Forest model on fresh data when drift is detected.

Process:
1. Extract recent normal BGP data (last 7 days)
2. Train new Isolation Forest model
3. Validate new model performance
4. Hot-swap: Replace old model with new one
5. Continue detection without downtime
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict
import psycopg2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class IFRetrainer:
    """Retrain Isolation Forest model"""
    
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
        self.model_dir = "model_artifacts"
        self.history_dir = f"{self.model_dir}/history"
        os.makedirs(self.history_dir, exist_ok=True)
        
        # New timestamped model paths
        self.new_model_path = f"{self.history_dir}/iso_forest_{self.timestamp}.pkl"
        self.new_metadata_path = f"{self.history_dir}/iso_forest_{self.timestamp}_metadata.json"
        
        # Production model paths (for hot-swap)
        self.prod_model_path = f"{self.model_dir}/iso_forest_bgp_production.pkl"
        self.prod_metadata_path = f"{self.model_dir}/iso_forest_metadata.json"
        
        # Training config
        self.training_days = 7
        self.contamination = 0.01
        self.n_estimators = 200
        self.max_samples = 5000
        self.training_start_time = None
        self.training_end_time = None
        
    def extract_training_data(self) -> pd.DataFrame:
        """Extract recent BGP features for training"""
        logger.info(f"Extracting training data from last {self.training_days} days...")
        
        conn = psycopg2.connect(**self.db_config)
        
        # Get features data (features table, not raw_bgp_data)
        query = f"""
            SELECT 
                announcements, withdrawals, total_updates,
                withdrawal_ratio, flap_count, path_length,
                unique_peers, message_rate, session_resets
            FROM features
            WHERE timestamp > NOW() - INTERVAL '{self.training_days} days'
            LIMIT 10000;
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"✅ Extracted {len(df)} samples")
        return df
    
    def train_model(self, X: np.ndarray) -> IsolationForest:
        """Train new Isolation Forest model"""
        logger.info("Training new Isolation Forest model...")
        
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=min(self.max_samples, len(X)),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X)
        
        logger.info(f"✅ Model trained with {self.n_estimators} trees")
        return model
    
    def validate_model(self, model: IsolationForest, X: np.ndarray) -> Dict:
        """Validate model performance"""
        logger.info("Validating new model...")
        
        # Get anomaly scores
        scores = model.decision_function(X)
        predictions = model.predict(X)
        
        anomaly_rate = (predictions == -1).mean()
        
        metrics = {
            'anomaly_rate': float(anomaly_rate),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'score_range': [float(scores.min()), float(scores.max())]
        }
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   Anomaly rate: {anomaly_rate:.2%}")
        logger.info(f"   Mean score: {metrics['mean_score']:.3f}")
        
        return metrics
    
    def hot_swap_model(self):
        """Replace old model with new model (hot swap)"""
        logger.info("Performing hot swap...")
        
        # Backup current production model
        if os.path.exists(self.prod_model_path):
            import shutil
            backup_path = f"{self.model_dir}/iso_forest_backup_{self.timestamp}.pkl"
            shutil.copy2(self.prod_model_path, backup_path)
            logger.info(f"✅ Production model backed up: {backup_path}")
        
        # Copy new model to production
        import shutil
        shutil.copy2(self.new_model_path, self.prod_model_path)
        logger.info(f"✅ New model activated: {self.prod_model_path}")
        
        # Copy metadata to production
        if os.path.exists(self.new_metadata_path):
            shutil.copy2(self.new_metadata_path, self.prod_metadata_path)
            logger.info(f"✅ Metadata updated")
        
        # Remove retraining flag if exists
        flag_path = f"{self.model_dir}/retrain_isolation_forest.flag"
        if os.path.exists(flag_path):
            os.remove(flag_path)
            logger.info("✅ Retraining flag removed")
    
    def save_metadata(self, metrics: Dict, training_info: dict) -> dict:
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
        drift_files = [f for f in os.listdir('model_artifacts') if f.startswith('drift_report_isolation_forest_')]
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
            'model_name': 'isolation_forest',
            'model_file': f'iso_forest_{self.timestamp}.pkl',
            'retrained_at': self.training_start_time.isoformat(),
            'training_completed_at': self.training_end_time.isoformat(),
            'training_duration_seconds': int((self.training_end_time - self.training_start_time).total_seconds()),
            'training_data': {
                'samples': training_info['samples'],
                'date_range_start': training_info['date_range'][0],
                'date_range_end': training_info['date_range'][1],
                'training_days': self.training_days
            },
            'model_config': {
                'type': 'IsolationForest',
                'n_estimators': self.n_estimators,
                'contamination': self.contamination,
                'max_samples': self.max_samples,
                'random_state': 42
            },
            'performance_metrics': {
                'anomaly_rate': metrics['anomaly_rate'],
                'mean_score': metrics['mean_score'],
                'std_score': metrics['std_score'],
                'score_range_min': metrics['score_range'][0],
                'score_range_max': metrics['score_range'][1]
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
        logger.info("ISOLATION FOREST RETRAINING")
        logger.info("="*60)
        
        self.training_start_time = datetime.now()
        
        try:
            # 1. Extract data
            df = self.extract_training_data()
            
            if len(df) < 100:
                logger.error("❌ Insufficient data for retraining")
                return False
            
            # Store date range for metadata
            date_range = (df.index[0].isoformat() if hasattr(df.index[0], 'isoformat') else 'unknown',
                         df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else 'unknown')
            
            # 2. Prepare features
            X = df.values
            
            # 3. Train new model
            new_model = self.train_model(X)
            
            # 4. Validate model
            metrics = self.validate_model(new_model, X)
            
            self.training_end_time = datetime.now()
            
            # 5. Save new model with timestamp
            joblib.dump(new_model, self.new_model_path)
            logger.info(f"✅ New model saved: {self.new_model_path}")
            
            # 6. Save comprehensive metadata
            training_info = {
                'samples': len(df),
                'date_range': date_range
            }
            metadata = self.save_metadata(metrics, training_info)
            
            # 7. Hot swap
            self.hot_swap_model()
            
            # 8. Update database drift status
            self.update_drift_status_db()
            
            logger.info("="*60)
            logger.info("✅ RETRAINING COMPLETE")
            logger.info("="*60)
            logger.info(f"Model file: {os.path.basename(self.new_model_path)}")
            logger.info(f"Metadata: {os.path.basename(self.new_metadata_path)}")
            logger.info(f"Training duration: {metadata['training_duration_seconds']}s")
            logger.info("Isolation Forest detector will automatically load new model")
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
                WHERE model_name = 'isolation_forest';
            """, (self.training_end_time,))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ Database drift status updated")
        except Exception as e:
            logger.warning(f"⚠️ Could not update database: {e}")


if __name__ == "__main__":
    retrainer = IFRetrainer()
    success = retrainer.run_retraining()
    sys.exit(0 if success else 1)
