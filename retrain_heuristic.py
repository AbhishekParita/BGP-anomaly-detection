"""
Heuristic Rules Retraining Script
==================================
Updates heuristic thresholds based on fresh data when drift is detected.

Process:
1. Extract recent BGP feature statistics (last 7 days)
2. Calculate new optimal thresholds (95th percentile)
3. Validate threshold effectiveness
4. Hot-swap: Update rules configuration
5. Continue detection without downtime
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import psycopg2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class HeuristicRetrainer:
    """Update heuristic detection thresholds"""
    
    def __init__(self):
        # Database config
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Config paths
        self.config_dir = "model_artifacts"
        self.old_config_path = f"{self.config_dir}/heuristic_rules.json"
        self.new_config_path = f"{self.config_dir}/heuristic_rules_new.json"
        self.backup_path = f"{self.config_dir}/heuristic_rules_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Training config
        self.training_days = 7
        self.percentile = 95  # Use 95th percentile as threshold
        
        # Default thresholds (fallback)
        self.default_rules = {
            'total_updates_threshold': 200,
            'withdrawal_ratio_threshold': 0.3,
            'flap_count_threshold': 50,
            'path_length_threshold': 8,
            'message_rate_threshold': 100,
            'session_resets_threshold': 3
        }
        
    def extract_training_data(self) -> pd.DataFrame:
        """Extract recent BGP features for threshold calculation"""
        logger.info(f"Extracting data from last {self.training_days} days...")
        
        conn = psycopg2.connect(**self.db_config)
        
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
    
    def calculate_thresholds(self, df: pd.DataFrame) -> dict:
        """Calculate new thresholds based on data percentiles"""
        logger.info(f"Calculating thresholds ({self.percentile}th percentile)...")
        
        rules = {
            'total_updates_threshold': float(np.percentile(df['total_updates'], self.percentile)),
            'withdrawal_ratio_threshold': float(np.percentile(df['withdrawal_ratio'], self.percentile)),
            'flap_count_threshold': float(np.percentile(df['flap_count'], self.percentile)),
            'path_length_threshold': float(np.percentile(df['path_length'], self.percentile)),
            'message_rate_threshold': float(np.percentile(df['message_rate'], self.percentile)),
            'session_resets_threshold': float(np.percentile(df['session_resets'], self.percentile))
        }
        
        logger.info("✅ New thresholds calculated:")
        for key, value in rules.items():
            logger.info(f"   {key}: {value:.2f}")
        
        return rules
    
    def validate_thresholds(self, df: pd.DataFrame, rules: dict) -> dict:
        """Validate threshold effectiveness"""
        logger.info("Validating new thresholds...")
        
        # Apply rules to data
        total_samples = len(df)
        
        violations = {
            'total_updates': int((df['total_updates'] > rules['total_updates_threshold']).sum()),
            'withdrawal_ratio': int((df['withdrawal_ratio'] > rules['withdrawal_ratio_threshold']).sum()),
            'flap_count': int((df['flap_count'] > rules['flap_count_threshold']).sum()),
            'path_length': int((df['path_length'] > rules['path_length_threshold']).sum()),
            'message_rate': int((df['message_rate'] > rules['message_rate_threshold']).sum()),
            'session_resets': int((df['session_resets'] > rules['session_resets_threshold']).sum())
        }
        
        # Calculate violation rates
        violation_rates = {k: float(v / total_samples) for k, v in violations.items()}
        
        # Any violation = anomaly
        any_violation = int(sum(
            (df['total_updates'] > rules['total_updates_threshold']) |
            (df['withdrawal_ratio'] > rules['withdrawal_ratio_threshold']) |
            (df['flap_count'] > rules['flap_count_threshold']) |
            (df['path_length'] > rules['path_length_threshold']) |
            (df['message_rate'] > rules['message_rate_threshold']) |
            (df['session_resets'] > rules['session_resets_threshold'])
        ))
        
        overall_anomaly_rate = any_violation / total_samples
        
        metrics = {
            'violation_counts': violations,
            'violation_rates': violation_rates,
            'overall_anomaly_rate': float(overall_anomaly_rate),
            'total_samples': int(total_samples)
        }
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   Overall anomaly rate: {overall_anomaly_rate:.2%}")
        for key, rate in violation_rates.items():
            logger.info(f"   {key}: {rate:.2%}")
        
        return metrics
    
    def hot_swap_config(self):
        """Replace old config with new config (hot swap)"""
        logger.info("Performing hot swap...")
        
        # Backup old config
        if os.path.exists(self.old_config_path):
            import shutil
            shutil.copy2(self.old_config_path, self.backup_path)
            logger.info(f"✅ Old config backed up: {self.backup_path}")
        
        # Replace with new config
        import shutil
        shutil.move(self.new_config_path, self.old_config_path)
        logger.info(f"✅ New config activated: {self.old_config_path}")
        
        # Remove retraining flag
        flag_path = f"{self.config_dir}/retrain_heuristic.flag"
        if os.path.exists(flag_path):
            os.remove(flag_path)
            logger.info("✅ Retraining flag removed")
    
    def run_retraining(self):
        """Execute complete retraining process"""
        logger.info("="*60)
        logger.info("HEURISTIC RULES RETRAINING")
        logger.info("="*60)
        
        try:
            # 1. Extract data
            df = self.extract_training_data()
            
            if len(df) < 100:
                logger.error("❌ Insufficient data for threshold calculation")
                logger.info("Using default thresholds...")
                rules = self.default_rules
            else:
                # 2. Calculate new thresholds
                rules = self.calculate_thresholds(df)
                
                # 3. Validate thresholds
                metrics = self.validate_thresholds(df, rules)
            
            # 4. Save new config
            config = {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(df),
                'training_days': self.training_days,
                'percentile': self.percentile,
                'rules': rules
            }
            
            if len(df) >= 100:
                config['validation_metrics'] = metrics
            
            with open(self.new_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ New config saved: {self.new_config_path}")
            
            # 5. Hot swap
            self.hot_swap_config()
            
            logger.info("="*60)
            logger.info("✅ RETRAINING COMPLETE")
            logger.info("="*60)
            logger.info("Heuristic detector will automatically load new rules")
            logger.info("on next detection cycle (within 10 seconds)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Create config directory if needed
    os.makedirs("model_artifacts", exist_ok=True)
    
    retrainer = HeuristicRetrainer()
    success = retrainer.run_retraining()
    sys.exit(0 if success else 1)
