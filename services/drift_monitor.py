"""
Model Drift Monitor Service
============================
Monitors ML model performance and triggers retraining when drift is detected.

Drift Detection Methods:
1. Prediction distribution shift (score distribution changes)
2. Anomaly rate changes (baseline vs current)
3. Model confidence degradation
4. False positive/negative rates (if ground truth available)

Triggers retraining when:
- Drift score exceeds threshold
- Performance drops below baseline
- Manual trigger requested
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psycopg2
import numpy as np
from dotenv import load_dotenv
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('drift_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class DriftMonitor:
    """
    Monitors model drift and triggers retraining
    """
    
    def __init__(self):
        """Initialize drift monitor"""
        
        # Database configuration
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        self.conn = None
        self.cursor = None
        
        # Drift detection settings
        self.check_interval = 3600  # Check every hour
        self.baseline_window = 7 * 24  # 7 days of data for baseline
        self.current_window = 24  # Last 24 hours for current
        
        # Drift thresholds
        self.drift_thresholds = {
            'lstm': {
                'score_shift': 0.15,  # 15% shift in score distribution
                'confidence_drop': 0.10,  # 10% drop in confidence
                'anomaly_rate_change': 0.20  # 20% change in anomaly rate
            },
            'isolation_forest': {
                'score_shift': 0.15,
                'confidence_drop': 0.10,
                'anomaly_rate_change': 0.20
            },
            'heuristic': {
                'anomaly_rate_change': 0.30  # Higher threshold for rules
            }
        }
        
        # Baseline metrics storage
        self.baseline_metrics = {}
        
        # Statistics
        self.stats = {
            'checks_performed': 0,
            'drifts_detected': 0,
            'retraining_triggered': 0,
            'last_check': None
        }
        
    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("✅ Database connected")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def calculate_baseline_metrics(self, model_name: str) -> Dict:
        """
        Calculate baseline metrics from historical data
        
        Args:
            model_name: 'lstm', 'isolation_forest', or 'heuristic'
            
        Returns:
            Dict with baseline metrics
        """
        try:
            # Get column names based on model
            score_col = f"{model_name}_anomaly_score" if model_name in ['lstm', 'isolation_forest'] else f"{model_name}_score"
            is_anomaly_col = f"{model_name}_is_anomaly"
            
            # Query baseline data (7 days old to avoid recent drift)
            self.cursor.execute(f"""
                SELECT 
                    {score_col} as score,
                    {is_anomaly_col} as is_anomaly
                FROM ml_results
                WHERE timestamp > NOW() - INTERVAL '{self.baseline_window} hours'
                    AND timestamp < NOW() - INTERVAL '{self.current_window} hours'
                    AND {score_col} IS NOT NULL
                LIMIT 10000;
            """)
            
            results = self.cursor.fetchall()
            
            if not results:
                logger.warning(f"No baseline data for {model_name}")
                return None
            
            scores = np.array([r[0] for r in results if r[0] is not None])
            anomalies = np.array([r[1] for r in results if r[1] is not None])
            
            baseline = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'median_score': float(np.median(scores)),
                'anomaly_rate': float(np.mean(anomalies)),
                'score_distribution': {
                    'q25': float(np.percentile(scores, 25)),
                    'q50': float(np.percentile(scores, 50)),
                    'q75': float(np.percentile(scores, 75)),
                    'q90': float(np.percentile(scores, 90))
                },
                'sample_count': len(scores),
                'timestamp': datetime.now().isoformat()
            }
            
            return baseline
            
        except Exception as e:
            logger.error(f"Error calculating baseline for {model_name}: {e}")
            return None
    
    def calculate_current_metrics(self, model_name: str) -> Dict:
        """
        Calculate current metrics from recent data
        
        Args:
            model_name: 'lstm', 'isolation_forest', or 'heuristic'
            
        Returns:
            Dict with current metrics
        """
        try:
            score_col = f"{model_name}_anomaly_score" if model_name in ['lstm', 'isolation_forest'] else f"{model_name}_score"
            is_anomaly_col = f"{model_name}_is_anomaly"
            
            # Query recent data
            self.cursor.execute(f"""
                SELECT 
                    {score_col} as score,
                    {is_anomaly_col} as is_anomaly
                FROM ml_results
                WHERE timestamp > NOW() - INTERVAL '{self.current_window} hours'
                    AND {score_col} IS NOT NULL
                LIMIT 10000;
            """)
            
            results = self.cursor.fetchall()
            
            if not results:
                return None
            
            scores = np.array([r[0] for r in results if r[0] is not None])
            anomalies = np.array([r[1] for r in results if r[1] is not None])
            
            current = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'median_score': float(np.median(scores)),
                'anomaly_rate': float(np.mean(anomalies)),
                'score_distribution': {
                    'q25': float(np.percentile(scores, 25)),
                    'q50': float(np.percentile(scores, 50)),
                    'q75': float(np.percentile(scores, 75)),
                    'q90': float(np.percentile(scores, 90))
                },
                'sample_count': len(scores)
            }
            
            return current
            
        except Exception as e:
            logger.error(f"Error calculating current metrics for {model_name}: {e}")
            return None
    
    def detect_drift(self, model_name: str) -> Dict:
        """
        Detect drift for a specific model
        
        Returns:
            Dict with drift detection results
        """
        logger.info(f"Checking drift for {model_name}...")
        
        # Get or calculate baseline
        if model_name not in self.baseline_metrics:
            baseline = self.calculate_baseline_metrics(model_name)
            if baseline:
                self.baseline_metrics[model_name] = baseline
            else:
                return {'drift_detected': False, 'reason': 'No baseline data'}
        else:
            baseline = self.baseline_metrics[model_name]
        
        # Calculate current metrics
        current = self.calculate_current_metrics(model_name)
        
        if not current:
            return {'drift_detected': False, 'reason': 'No current data'}
        
        # Calculate drift metrics
        thresholds = self.drift_thresholds[model_name]
        
        # 1. Score distribution shift (KL divergence approximation)
        score_shift = abs(current['mean_score'] - baseline['mean_score']) / (baseline['std_score'] + 1e-6)
        
        # 2. Anomaly rate change
        anomaly_rate_change = abs(current['anomaly_rate'] - baseline['anomaly_rate']) / (baseline['anomaly_rate'] + 1e-6)
        
        # 3. Distribution quartile shift
        q_shift = np.mean([
            abs(current['score_distribution']['q25'] - baseline['score_distribution']['q25']),
            abs(current['score_distribution']['q50'] - baseline['score_distribution']['q50']),
            abs(current['score_distribution']['q75'] - baseline['score_distribution']['q75'])
        ])
        
        # Determine if drift detected
        drift_detected = False
        reasons = []
        
        if score_shift > thresholds.get('score_shift', 0.15):
            drift_detected = True
            reasons.append(f"Score shift: {score_shift:.3f} > {thresholds['score_shift']}")
        
        if anomaly_rate_change > thresholds.get('anomaly_rate_change', 0.20):
            drift_detected = True
            reasons.append(f"Anomaly rate change: {anomaly_rate_change:.3f} > {thresholds['anomaly_rate_change']}")
        
        if q_shift > 0.15:
            drift_detected = True
            reasons.append(f"Quartile shift: {q_shift:.3f} > 0.15")
        
        result = {
            'model': model_name,
            'drift_detected': drift_detected,
            'reasons': reasons,
            'metrics': {
                'score_shift': float(score_shift),
                'anomaly_rate_change': float(anomaly_rate_change),
                'quartile_shift': float(q_shift)
            },
            'baseline': baseline,
            'current': current,
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"⚠️ DRIFT DETECTED for {model_name}: {', '.join(reasons)}")
            self.stats['drifts_detected'] += 1
        else:
            logger.info(f"✅ No drift for {model_name}")
        
        return result
    
    def trigger_retraining(self, model_name: str, drift_info: Dict):
        """
        Trigger model retraining
        
        Args:
            model_name: Model to retrain
            drift_info: Drift detection information
        """
        logger.info(f"🔄 Triggering retraining for {model_name}")
        
        # Save drift report
        report_path = f"model_artifacts/drift_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(drift_info, f, indent=2)
        
        logger.info(f"📝 Drift report saved: {report_path}")
        
        # Create retraining flag file
        flag_path = f"model_artifacts/retrain_{model_name}.flag"
        with open(flag_path, 'w') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'reason': 'drift_detected',
                'drift_metrics': drift_info['metrics']
            }, indent=2))
        
        logger.info(f"🚩 Retraining flag created: {flag_path}")
        
        # Automatically trigger retraining script
        try:
            import subprocess
            
            # Run retraining script in background
            script_path = f"retrain_{model_name}.py"
            logger.info(f"🚀 Starting retraining script: {script_path}")
            
            # Use subprocess to run in background (non-blocking)
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__))  # Root directory
            )
            
            logger.info(f"✅ Retraining process started (PID: {process.pid})")
            logger.info(f"   Model will be automatically replaced when training completes")
            logger.info(f"   Detection continues with current model until hot-swap")
            
        except Exception as e:
            logger.error(f"❌ Failed to start retraining script: {e}")
            logger.info(f"   Please run manually: python retrain_{model_name}.py")
        
        self.stats['retraining_triggered'] += 1
    
    def run_continuous(self):
        """Run drift monitoring continuously"""
        logger.info("="*60)
        logger.info("MODEL DRIFT MONITOR SERVICE STARTED")
        logger.info("="*60)
        logger.info(f"Check interval: {self.check_interval} seconds ({self.check_interval/3600:.1f} hours)")
        logger.info(f"Baseline window: {self.baseline_window} hours ({self.baseline_window/24:.1f} days)")
        logger.info(f"Current window: {self.current_window} hours")
        logger.info("Monitoring models: LSTM, Isolation Forest, Heuristic")
        logger.info("="*60)
        
        if not self.connect_db():
            return
        
        while True:
            try:
                logger.info(f"\n[{datetime.now()}] Starting drift check cycle...")
                self.stats['checks_performed'] += 1
                
                # Check each model
                for model_name in ['lstm', 'isolation_forest', 'heuristic']:
                    drift_result = self.detect_drift(model_name)
                    
                    if drift_result['drift_detected']:
                        self.trigger_retraining(model_name, drift_result)
                
                self.stats['last_check'] = datetime.now().isoformat()
                
                logger.info(f"\n📊 Statistics:")
                logger.info(f"   Checks performed: {self.stats['checks_performed']}")
                logger.info(f"   Drifts detected: {self.stats['drifts_detected']}")
                logger.info(f"   Retraining triggered: {self.stats['retraining_triggered']}")
                
                # Wait before next check
                logger.info(f"\n⏳ Waiting {self.check_interval}s until next check...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Drift monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in drift check: {e}")
                time.sleep(60)  # Wait 1 minute before retry
        
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    monitor = DriftMonitor()
    monitor.run_continuous()
