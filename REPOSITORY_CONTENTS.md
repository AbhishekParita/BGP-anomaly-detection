# 📦 GitHub Repository Contents

## ✅ Files INCLUDED in Git (will be pushed to GitHub)

### 🐍 Core Service Files
- `ris_live_collector.py` - Collects live BGP data from RIS
- `feature_aggregator.py` - Extracts statistical features
- `lstm_detector.py` - LSTM anomaly detection
- `if_detector.py` - Isolation Forest detection
- `heuristic_detector.py` - Rule-based detection
- `ensemble_coordinator.py` - Ensemble voting system
- `correlation_engine.py` - Alert correlation
- `drift_monitor.py` - Model drift monitoring

### 🌐 API & Dashboard
- `api.py` - FastAPI REST API server
- `dashboard_enhanced.html` - Real-time monitoring dashboard

### 🔧 Configuration Templates (Safe for GitHub)
- `config.template.py` - Configuration template (NO PASSWORDS)
- `db_connector.template.py` - Database connection template (NO PASSWORDS)

### 📚 Documentation
- `README.md` - Main GitHub documentation
- `START_SYSTEM.md` - Complete startup guide
- `DRIFT_DETECTION_EXPLAINED.md` - Drift algorithm details
- `SYSTEM_THRESHOLDS.json` - Threshold reference
- `GITHUB_SETUP.md` - GitHub setup instructions
- `LICENSE` - MIT License

### 🛠️ Utility Scripts
- `run_all_services.py` - Automated service management
- `run_training.py` - LSTM training script
- `train_if_model.py` - Isolation Forest training
- `evaluate_anomaly_detection.py` - Model evaluation
- `test_new_data.py` - Testing with custom data
- `visualize_results.py` - Results visualization
- `anomaly_pipeline.py` - Pipeline utilities
- `bgp_lstm_pipeline.py` - BGP LSTM pipeline
- `bmp_generator.py` - BMP data generator
- `stream_generator.py` - Stream testing
- `datasetfilter.py` - Dataset filtering
- `hybrid_detector.py` - Hybrid detection

### 🗄️ Database
- `schema_functions.sql` - PostgreSQL schema
- `requirements.txt` - Python dependencies

### 🚫 Configuration
- `.gitignore` - Git ignore rules

### 📁 BGP Generator (if needed)
- `BGP_generator/fast_training_data.py`
- `BGP_generator/generate_training_data.py`

## ❌ Files EXCLUDED from Git (in .gitignore)

### 🔒 Private/Sensitive Files
- `config.py` - Contains PostgreSQL password and credentials
- `db_connector.py` - Contains database credentials
- `.env` files - Environment variables

### 📊 Data Files (Large/Generated)
- `raw_data/` - Raw data files (code_red.csv, nimda.csv, slammer.csv)
- `data/` - Training data (synthetic_30d.csv, etc.)
- `datasetfiles/` - Test datasets (*.csv files)
- `model_artifacts/` - Trained model files
- `model_output/` - Model outputs and weights
- `testing_run/` - Test results
- `ensemble_plots/` - Generated plots
- `anomaly_detection_evaluation/` - Evaluation results

### 🐍 Python Generated Files
- `__pycache__/` - Python cache
- `*.pyc`, `*.pyo`, `*.pyd` - Compiled Python
- `venv_wsl/` - Virtual environment
- `venv/`, `env/` - Other virtual environments

### 📦 External Dependencies
- `kafka_2.13-3.9.0/` - Kafka installation
- `*.tgz`, `*.tgz.*` - Archive files
- `routinator/` - Routinator files

### 🗃️ Generated/Temporary Files
- `logs/`, `*.log` - Log files
- `.vscode/`, `.idea/` - IDE settings
- `.DS_Store`, `Thumbs.db` - OS files
- `*.csv` - CSV data files
- `*.pkl` - Pickle files
- `*.h5`, `*.weights.h5` - Model weights

## 📊 Repository Statistics

**Total files to be pushed:** ~30-40 Python files + documentation + templates
**Total size:** ~500KB-1MB (without data/models/venv)

**Users will need to provide:**
1. PostgreSQL credentials (in config.py)
2. Train their own models OR download pre-trained models separately
3. Install Kafka locally
4. Set up their own database

## 🔐 Security Verification

Before pushing, verify no sensitive data:

```bash
# Check config.py is ignored
git check-ignore config.py

# Check db_connector.py is ignored  
git check-ignore db_connector.py

# List all files that will be committed
git ls-files

# Search for passwords in tracked files (should find NONE except templates)
git grep -i "password" -- ':!*.template.py'
```

## ✅ Ready to Push!

All sensitive information is protected. Only code, documentation, and templates will be pushed to GitHub.
