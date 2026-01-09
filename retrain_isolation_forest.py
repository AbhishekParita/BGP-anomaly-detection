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
        
        # Model paths
        self.model_dir = "model_artifacts"
        self.old_model_path = f"{self.model_dir}/iso_forest_bgp_production.pkl"
        self.new_model_path = f"{self.model_dir}/iso_forest_bgp_production_new.pkl"
        self.backup_path = f"{self.model_dir}/iso_forest_bgp_production_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Training config
        self.training_days = 7
        self.contamination = 0.01
        self.n_estimators = 200
        self.max_samples = 5000
        
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
        flag_path = f"{self.model_dir}/retrain_isolation_forest.flag"
        if os.path.exists(flag_path):
            os.remove(flag_path)
            logger.info("✅ Retraining flag removed")
    
    def run_retraining(self):
        """Execute complete retraining process"""
        logger.info("="*60)
        logger.info("ISOLATION FOREST RETRAINING")
        logger.info("="*60)
        
        try:
            # 1. Extract data
            df = self.extract_training_data()
            
            if len(df) < 100:
                logger.error("❌ Insufficient data for retraining")
                return False
            
            # 2. Prepare features
            X = df.values
            
            # 3. Train new model
            new_model = self.train_model(X)
            
            # 4. Validate model
            metrics = self.validate_model(new_model, X)
            
            # 5. Save new model
            joblib.dump(new_model, self.new_model_path)
            logger.info(f"✅ New model saved: {self.new_model_path}")
            
            # 6. Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(df),
                'training_days': self.training_days,
                'n_estimators': self.n_estimators,
                'contamination': self.contamination,
                'validation_metrics': metrics
            }
            
            with open(f"{self.model_dir}/iso_forest_metadata_new.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 7. Hot swap
            self.hot_swap_model()
            
            logger.info("="*60)
            logger.info("✅ RETRAINING COMPLETE")
            logger.info("="*60)
            logger.info("Isolation Forest detector will automatically load new model")
            logger.info("on next detection cycle (within 10 seconds)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            return False


if __name__ == "__main__":
    retrainer = IFRetrainer()
    success = retrainer.run_retraining()
    sys.exit(0 if success else 1)
