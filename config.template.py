"""
Configuration Template for BGP Anomaly Detection System
Copy this file to config.py and fill in your actual values
"""

# ==================== DATABASE CONFIGURATION ====================
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'your_database_name',
    'user': 'your_username',
    'password': 'your_password'  # CHANGE THIS
}

# ==================== ALERT THRESHOLDS ====================
# Anomaly score thresholds for different severity levels
CRITICAL_THRESHOLD = 2.54
HIGH_THRESHOLD = 2.14
MEDIUM_THRESHOLD = 1.82
LOW_THRESHOLD = 1.58

# ==================== DATABASE LIMITS ====================
MAX_DATABASE_RECORDS = 100000  # Maximum records to keep in database
RETENTION_DAYS = 30  # Days to retain historical data
MAX_RECORDS = 50000  # Maximum records per query

# ==================== KAFKA CONFIGURATION ====================
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPICS = {
    'raw_bgp': 'bgp-raw-updates',
    'features': 'bgp-features',
    'lstm_detections': 'lstm-detections',
    'if_detections': 'if-detections',
    'heuristic_detections': 'heuristic-detections',
    'ensemble_alerts': 'ensemble-alerts',
    'correlated_alerts': 'correlated-alerts',
    'drift_reports': 'drift-reports'
}

# ==================== MODEL PATHS ====================
MODEL_PATHS = {
    'lstm': './model_output/lstm_model_for_pkl.weights.h5',
    'if': './model_output/isolation_forest.pkl',
    'config': './model_output/config.json'
}

# ==================== STREAM PROCESSING ====================
STREAM_BATCH_SIZE = 100
STREAM_TIMEOUT = 5.0  # seconds

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== API CONFIGURATION ====================
API_HOST = '0.0.0.0'
API_PORT = 8000
API_CORS_ORIGINS = ['*']

# ==================== DRIFT DETECTION ====================
DRIFT_CHECK_INTERVAL = 3600  # Check every hour (in seconds)
DRIFT_BASELINE_WINDOW = 7  # Days for baseline calculation
DRIFT_CURRENT_WINDOW = 1  # Days for current period

# Drift thresholds
DRIFT_THRESHOLDS = {
    'lstm': {
        'score_shift': 0.15,
        'anomaly_rate_change': 0.20
    },
    'isolation_forest': {
        'score_shift': 0.15,
        'anomaly_rate_change': 0.20
    },
    'heuristic': {
        'anomaly_rate_change': 0.30
    }
}

# ==================== ENSEMBLE CONFIGURATION ====================
ENSEMBLE_VOTING = {
    'weights': {
        'lstm': 0.4,
        'isolation_forest': 0.3,
        'heuristic': 0.3
    },
    'min_agreement': 2  # Minimum models that must agree
}

# ==================== HEURISTIC RULES THRESHOLDS ====================
HEURISTIC_THRESHOLDS = {
    'withdrawal_ratio_high': 0.7,
    'update_storm': 1000,
    'path_length_max': 10,
    'path_length_min': 2,
    'prefix_instability': 100,
    'as_path_anomaly': 3
}

# ==================== SYSTEM PROTECTION ====================
MAX_CPU_PERCENT = 80
MAX_MEMORY_PERCENT = 80
HEALTH_CHECK_INTERVAL = 60  # seconds
