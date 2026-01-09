# 🛡️ BGP Anomaly Detection System

An intelligent, real-time BGP (Border Gateway Protocol) anomaly detection system using ensemble machine learning and advanced drift monitoring capabilities.

## 📋 Overview

This system monitors BGP routing updates in real-time and detects various types of anomalies including route hijacks, route leaks, BGP flapping, peer instability, and prefix hijacks. It uses a multi-model ensemble approach combining LSTM (Long Short-Term Memory), Isolation Forest, and rule-based heuristic detectors to achieve high accuracy and minimize false positives.

### Key Features

- **🔄 Real-Time Processing**: Live BGP data streaming from RIS (Routing Information Service)
- **🤖 Multi-Model Ensemble**: Combines LSTM, Isolation Forest, and Heuristic detectors
- **📊 Advanced Analytics**: Feature aggregation with statistical analysis
- **🎯 Intelligent Correlation**: Context-aware alert correlation engine
- **📈 Model Drift Monitoring**: Automatic detection of model performance degradation
- **🖥️ Interactive Dashboard**: Real-time visualization with dark theme UI
- **🔔 Severity-Based Alerting**: Multi-level alert classification (Critical/High/Medium/Low)
- **🗄️ PostgreSQL Integration**: Persistent storage with efficient querying

## 🏗️ Architecture

The system follows a microservices architecture with 8 independent services:

```
┌─────────────────┐
│  RIS Live       │  Collects live BGP updates
│  Collector      │
└────────┬────────┘
         │ Kafka: bgp-raw-updates
         ↓
┌─────────────────┐
│  Feature        │  Extracts statistical features
│  Aggregator     │
└────────┬────────┘
         │ Kafka: bgp-features
         ├──────────┬──────────┐
         ↓          ↓          ↓
┌─────────────┐ ┌──────────┐ ┌─────────────┐
│ LSTM        │ │Isolation │ │ Heuristic   │
│ Detector    │ │Forest    │ │ Detector    │
└──────┬──────┘ └────┬─────┘ └──────┬──────┘
       │             │              │
       └──────┬──────┴──────┬───────┘
              │ Kafka: detections
              ↓
       ┌──────────────┐
       │  Ensemble     │  Voting & fusion
       │  Coordinator  │
       └──────┬────────┘
              │ Kafka: ensemble-alerts
              ↓
       ┌──────────────┐
       │  Correlation │  Context analysis
       │  Engine      │
       └──────┬────────┘
              │ Kafka: correlated-alerts
              ├──────────┬────────────┐
              ↓          ↓            ↓
       ┌──────────┐  ┌─────────┐  ┌──────────┐
       │PostgreSQL│  │ Drift   │  │   API    │
       │ Database │  │ Monitor │  │ Server   │
       └──────────┘  └─────────┘  └────┬─────┘
                                        ↓
                                  ┌──────────┐
                                  │Dashboard │
                                  └──────────┘
```

## 🔍 Detection Methods

### 1. **LSTM (Long Short-Term Memory)**
- Deep learning model for temporal pattern analysis
- Captures long-term dependencies in BGP behavior
- Trained on 30 days of normal BGP traffic
- **Threshold**: Anomaly score > 2.54 (Critical)

### 2. **Isolation Forest**
- Unsupervised anomaly detection algorithm
- Efficient for high-dimensional feature spaces
- Identifies outliers based on isolation properties
- **Threshold**: Anomaly score > 2.14 (High)

### 3. **Heuristic Rules**
- Domain-specific BGP anomaly patterns
- 6 rule categories:
  - **Withdrawal Ratio**: Excessive route withdrawals (>70%)
  - **Update Storm**: Rapid update bursts (>1000 updates)
  - **AS Path Length**: Abnormal path lengths (>10 or <2 hops)
  - **Prefix Instability**: Frequent prefix flapping (>100 changes)
  - **AS Path Anomaly**: Suspicious path patterns (>3 anomalies)

### 4. **Ensemble Voting**
- Weighted majority voting (LSTM: 40%, IF: 30%, Heuristic: 30%)
- Requires minimum 2 models agreement
- Confidence score calculation

## 📦 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 17.7+
- Apache Kafka 2.13-3.9.0
- 4GB+ RAM
- Ubuntu/Debian or Windows 10/11

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/bgp-anomaly-detection.git
cd bgp-anomaly-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Database

1. Install PostgreSQL and create database:
```sql
CREATE DATABASE bgp_anomaly_detection;
```

2. Run schema creation:
```bash
psql -U postgres -d bgp_anomaly_detection -f schema_functions.sql
```

### Step 5: Configure Application

1. Copy configuration template:
```bash
cp config.template.py config.py
```

2. Edit `config.py` with your settings:
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bgp_anomaly_detection',
    'user': 'your_username',
    'password': 'your_password'
}
```

### Step 6: Setup Kafka

1. Download and extract Kafka (if not already installed)
2. Start Zookeeper:
```bash
# Linux/Mac
bin/zookeeper-server-start.sh config/zookeeper.properties

# Windows
bin\windows\zookeeper-server-start.bat config\zookeeper.properties
```

3. Start Kafka:
```bash
# Linux/Mac
bin/kafka-server-start.sh config/server.properties

# Windows
bin\windows\kafka-server-start.bat config\server.properties
```

## 🚀 Running the System

### Option 1: Automated Startup (Recommended)

```bash
python run_all_services.py start
```

This starts all 8 services in the correct order:
1. RIS Live Collector
2. Feature Aggregator
3. LSTM Detector
4. Isolation Forest Detector
5. Heuristic Detector
6. Ensemble Coordinator
7. Correlation Engine
8. Drift Monitor

### Option 2: Manual Startup

**Terminal 1 - Data Collection:**
```bash
python ris_live_collector.py
python feature_aggregator.py
```

**Terminal 2 - Detection Models:**
```bash
python lstm_detector.py
python if_detector.py
python heuristic_detector.py
```

**Terminal 3 - Ensemble & Monitoring:**
```bash
python ensemble_coordinator.py
python correlation_engine.py
python drift_monitor.py
```

**Terminal 4 - API & Dashboard:**
```bash
python api.py
# Open dashboard_enhanced.html in browser
```

### Monitoring Services

```bash
# Check service status
python run_all_services.py status

# Stop all services
python run_all_services.py stop
```

## 📊 Dashboard Access

1. Start the API server (automatically started with `run_all_services.py`)
2. Open `dashboard_enhanced.html` in your web browser
3. Dashboard URL: `http://localhost:8000` (or open the HTML file directly)

### Dashboard Features

- **📈 Live Statistics**: Total detections, hourly rate, model confidence
- **📉 Time Series Chart**: 24-hour detection trends
- **🎯 Model Performance**: Individual model accuracy metrics
- **🔔 Recent Alerts**: Last 10 detections with severity levels
- **⚠️ Drift Monitoring**: Real-time model drift status
- **🔄 Auto-refresh**: Updates every 10 seconds

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/live-stats` | GET | Current system statistics |
| `/api/time-series` | GET | Historical detection data |
| `/api/recent-detections` | GET | Latest anomaly alerts |
| `/api/drift-status` | GET | Model drift monitoring status |

## 📈 Model Training

### LSTM Model

```bash
python run_training.py
```

This script:
- Loads training data from `data/synthetic_30d.csv`
- Trains LSTM with bi-directional layers
- Saves model to `model_output/lstm_model_for_pkl.weights.h5`
- Generates evaluation metrics

### Isolation Forest

```bash
python train_if_model.py
```

Features:
- Contamination factor: 0.05 (5% anomaly rate)
- Auto-saves to `model_output/isolation_forest.pkl`

## 🔄 Drift Detection

The system automatically monitors model performance degradation using 3 methods:

### 1. **Score Shift Detection**
- Formula: `|median_current - median_baseline| / IQR_baseline`
- **LSTM/IF Threshold**: 0.15
- **Heuristic Threshold**: N/A

### 2. **Anomaly Rate Change**
- Formula: `|(rate_current - rate_baseline) / rate_baseline|`
- **LSTM/IF Threshold**: 0.20 (±20% change)
- **Heuristic Threshold**: 0.30 (±30% change)

### 3. **Quartile Shift Analysis**
- Compares Q1, Q2, Q3 between baseline (7 days) and current (24 hours)
- Alerts when distribution shifts significantly

**Automatic Retraining**: When drift is detected, the system flags affected models for retraining.

## 📖 Configuration Reference

All threshold values are documented in `SYSTEM_THRESHOLDS.json`:

- Alert severity levels
- Drift detection parameters
- Heuristic rule thresholds
- Time windows and intervals
- Database retention policies
- Ensemble voting weights

## 🧪 Testing

### Run with Test Data

```bash
# Test with specific anomaly types
python test_new_data.py --data datasetfiles/test_hijack.csv
python test_new_data.py --data datasetfiles/test_route_leak.csv
python test_new_data.py --data datasetfiles/test_flapping.csv
```

### Evaluation

```bash
python evaluate_anomaly_detection.py
```

Results saved to `anomaly_detection_evaluation/`:
- `evaluation_summary.json`: Metrics (precision, recall, F1-score)
- `anomaly_detection_results.csv`: Detailed predictions
- `detected_alerts.csv`: Alert classifications

## 🛠️ Troubleshooting

### Services Won't Start

1. Check PostgreSQL is running:
```bash
# Windows
pg_ctl status -D "C:\Program Files\PostgreSQL\17\data"

# Linux
sudo systemctl status postgresql
```

2. Verify Kafka is running:
```bash
# List topics to test connection
kafka-topics.sh --list --bootstrap-server localhost:9092
```

3. Check port availability:
```bash
# Windows
netstat -ano | findstr :8000
netstat -ano | findstr :9092

# Linux
netstat -tuln | grep 8000
netstat -tuln | grep 9092
```

### Drift Tables Missing

The API automatically creates drift tables on startup. If you see errors:

```bash
# Manually create tables
psql -U postgres -d bgp_anomaly_detection
CREATE TABLE IF NOT EXISTS drift_flags (...);
CREATE TABLE IF NOT EXISTS drift_reports (...);
```

### High Memory Usage

Adjust limits in `config.py`:
```python
MAX_DATABASE_RECORDS = 50000  # Reduce from 100000
RETENTION_DAYS = 7  # Reduce from 30
```

## 📚 Documentation Files

- `START_SYSTEM.md`: Complete startup instructions
- `DRIFT_DETECTION_EXPLAINED.md`: Drift algorithm details
- `SYSTEM_THRESHOLDS.json`: Threshold reference

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [RIPE RIS Live API](https://ris-live.ripe.net/)
- [BGP Anomaly Detection Research Papers](https://scholar.google.com/scholar?q=bgp+anomaly+detection)
- [LSTM for Time Series](https://arxiv.org/abs/1506.00019)
- [Isolation Forest Algorithm](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

## 👥 Authors

- **Your Name** - *Initial work*

## 🙏 Acknowledgments

- RIPE NCC for RIS Live streaming service
- TensorFlow/Keras team for deep learning framework
- Scikit-learn for machine learning algorithms
- FastAPI for API framework

---

**⭐ If you find this project useful, please consider giving it a star!**

For issues and feature requests, please use the [GitHub Issues](https://github.com/yourusername/bgp-anomaly-detection/issues) page.
