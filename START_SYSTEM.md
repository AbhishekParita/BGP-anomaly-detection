# 🚀 BGP Anomaly Detection System - Complete Startup Guide

## Prerequisites
- PostgreSQL database running (localhost:5432)
- Python virtual environment at: `E:\Advantal_models\lstm_model\venv`
- All services in `services/` directory

---

## 📋 TERMINAL 1: Main Services (8 Microservices)

### Navigate to project directory
```powershell
cd E:\Advantal_models\lstm_model
```

### Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Start all 8 services (Run each in sequence or use screen/tmux)

#### Service 1: RIS Live Collector (Data Ingestion)
```powershell
python services/ris_live_collector.py
```
**Purpose**: Connects to RIPE RIS Live WebSocket, receives real-time BGP updates
**Expected Output**: "Connected to RIS Live...", "Received BGP update from..."
**Leave Running**: Yes (continuous data collection)

---

## 📋 TERMINAL 2: Detection Pipeline Services

### Open new PowerShell terminal
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
```

#### Service 2: Feature Aggregator
```powershell
python services/feature_aggregator.py
```
**Purpose**: Aggregates BGP data into 30-second windows, extracts 9 features
**Expected Output**: "Processing batch...", "Aggregated X records"
**Leave Running**: Yes

---

## 📋 TERMINAL 3: ML Detection Models (3 Services)

### Open new PowerShell terminal
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
```

#### Service 3: LSTM Detector
```powershell
python services/lstm_detector.py
```
**Purpose**: Neural network-based anomaly detection
**Expected Output**: "LSTM model loaded", "Processing batch..."
**Leave Running**: Yes

#### Service 4: Isolation Forest Detector (Same terminal, or open new one)
```powershell
python services/isolation_forest_detector.py
```
**Purpose**: Statistical anomaly detection using Isolation Forest
**Expected Output**: "Isolation Forest model loaded", "Detected X anomalies"
**Leave Running**: Yes

#### Service 5: Heuristic Detector
```powershell
python services/heuristic_detector.py
```
**Purpose**: Rule-based anomaly detection (thresholds)
**Expected Output**: "Heuristic detector started", "Processing..."
**Leave Running**: Yes

---

## 📋 TERMINAL 4: Ensemble & Monitoring Services

### Open new PowerShell terminal
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
```

#### Service 6: Ensemble Coordinator
```powershell
python services/ensemble_coordinator.py
```
**Purpose**: Combines 3 model predictions using voting
**Expected Output**: "Ensemble coordinator started", "Voting result..."
**Leave Running**: Yes

#### Service 7: Correlation Engine
```powershell
python services/correlation_engine.py
```
**Purpose**: Correlates detections, reduces false positives
**Expected Output**: "Correlation engine started", "Processed alerts"
**Leave Running**: Yes

#### Service 8: Drift Monitor (Self-Healing)
```powershell
python services/drift_monitor.py
```
**Purpose**: Monitors model drift, triggers automatic retraining
**Expected Output**: "Drift monitor started", "Checking model health..."
**Leave Running**: Yes

---

## 📋 TERMINAL 5: API Server

### Open new PowerShell terminal
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
```

#### Start FastAPI Server
```powershell
python api.py
```
**Purpose**: REST API for dashboard and external access
**Expected Output**: 
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```
**API Endpoints**:
- http://localhost:8000/api/live-stats
- http://localhost:8000/api/time-series
- http://localhost:8000/api/recent-detections
- http://localhost:8000/api/drift-status

**Leave Running**: Yes

---

## 📋 DASHBOARD: Web Interface

### Open in Browser
```
file:///E:/Advantal_models/lstm_model/dashboard_enhanced.html
```

**OR** if you want to serve it:
```powershell
# In project directory
python -m http.server 8080
# Then open: http://localhost:8080/dashboard_enhanced.html
```

**Features**:
- Real-time statistics (10-second refresh)
- 4 interactive charts
- Model drift monitoring
- Recent detection events
- Dark theme

---

## 🎯 RECOMMENDED STARTUP SEQUENCE

### **Minimum Required (4 Terminals)**:

**Terminal 1**: Core Data Pipeline
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
python services/ris_live_collector.py
```

**Terminal 2**: Feature Processing + Detectors
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
python services/feature_aggregator.py
# Wait 10 seconds for it to stabilize, then Ctrl+C and start detectors
python services/lstm_detector.py &
python services/isolation_forest_detector.py &
python services/heuristic_detector.py
```

**Terminal 3**: Ensemble + Monitoring
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
python services/ensemble_coordinator.py &
python services/correlation_engine.py &
python services/drift_monitor.py
```

**Terminal 4**: API
```powershell
cd E:\Advantal_models\lstm_model
.\venv\Scripts\Activate.ps1
python api.py
```

**Browser**: Open dashboard
```
file:///E:/Advantal_models/lstm_model/dashboard_enhanced.html
```

---

## ✅ VERIFICATION CHECKLIST

After starting all services, verify:

### 1. Database Check
```powershell
psql -U postgres -d bgp_monitoring -c "SELECT COUNT(*) FROM bgp_messages;"
psql -U postgres -d bgp_monitoring -c "SELECT COUNT(*) FROM aggregated_features;"
psql -U postgres -d bgp_monitoring -c "SELECT COUNT(*) FROM ml_results;"
```
**Expected**: Numbers should be increasing (check twice with 30-second gap)

### 2. Service Status Check
Each terminal should show:
- No error messages
- "Processing..." or similar active messages
- Timestamps indicating recent activity

### 3. API Check
```powershell
curl http://localhost:8000/api/live-stats
```
**Expected**: JSON response with processing statistics

### 4. Dashboard Check
- Open dashboard in browser
- Should show live data
- Numbers should update every 10 seconds
- Drift monitoring section should show "Healthy" status
- Charts should display data

---

## 🛑 STOPPING THE SYSTEM

### Stop all services:
1. Press `Ctrl+C` in each terminal (one by one)
2. API will shut down gracefully
3. Database connections will close automatically

### Quick Kill (if services hang):
```powershell
# Kill all Python processes (use with caution)
Get-Process python | Stop-Process -Force
```

---

## 🔄 RESTART AFTER SYSTEM REBOOT

### 1. Start PostgreSQL
```powershell
# Usually starts automatically, but if not:
pg_ctl -D "C:\Program Files\PostgreSQL\17\data" start
```

### 2. Follow startup sequence above (4 terminals)

### 3. Verify system health (use checklist)

---

## 📊 MONITORING TIPS

### Check System Load:
```powershell
# Monitor CPU/Memory usage
Get-Process python | Select-Object Name, CPU, WorkingSet | Format-Table -AutoSize
```

### Check Database Size:
```powershell
psql -U postgres -d bgp_monitoring -c "\dt+"
```

### View Recent Logs:
Each service prints to stdout/stderr in its terminal

### Check Drift Status:
```powershell
psql -U postgres -d bgp_monitoring -c "SELECT * FROM drift_flags ORDER BY timestamp DESC LIMIT 5;"
```

---

## ⚠️ TROUBLESHOOTING

### Issue: "Connection refused" errors
**Solution**: 
1. Check PostgreSQL is running
2. Verify database exists: `psql -U postgres -l | grep bgp_monitoring`
3. Check connection string in services

### Issue: No data in dashboard
**Solution**:
1. Check RIS collector is running (Terminal 1)
2. Verify feature aggregator is processing (Terminal 2)
3. Check database has records: `SELECT COUNT(*) FROM bgp_messages;`

### Issue: Models not loading
**Solution**:
1. Check model files exist in `model_output/`
2. Verify paths in detector scripts
3. Check for training completed: `ls model_output/`

### Issue: API not responding
**Solution**:
1. Check Terminal 4 for error messages
2. Verify port 8000 not in use: `netstat -ano | findstr :8000`
3. Test with curl: `curl http://localhost:8000/api/live-stats`

---

## 📞 SYSTEM ARCHITECTURE REMINDER

```
RIS Live Stream → RIS Collector → Feature Aggregator → 3 Detectors (LSTM, IF, Heuristic)
                                                           ↓
                                                    Ensemble Coordinator
                                                           ↓
                                                    Correlation Engine
                                                           ↓
                                                       Database
                                                           ↓
                      API ← Dashboard               Drift Monitor
                                                    (Self-Healing)
```

---

## 🎓 NOTES

- **Data Latency**: 5-10 minutes behind real-time (normal for RIS Live)
- **Auto-Cleanup**: Database automatically maintains last 50,000 records per table
- **Auto-Retraining**: Drift monitor checks hourly, retrains models if drift detected
- **Hot-Swap**: Model updates happen without service restart (zero downtime)
- **Dashboard Refresh**: 10 seconds for stats, 30 seconds for drift status

---

## 📝 QUICK REFERENCE COMMANDS

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Start core services (Terminal 1-3)
python services/ris_live_collector.py
python services/feature_aggregator.py
python services/lstm_detector.py
python services/isolation_forest_detector.py
python services/heuristic_detector.py
python services/ensemble_coordinator.py
python services/correlation_engine.py
python services/drift_monitor.py

# Start API (Terminal 4)
python api.py

# Database quick check
psql -U postgres -d bgp_monitoring -c "SELECT COUNT(*) FROM ml_results WHERE timestamp > NOW() - INTERVAL '5 minutes';"

# View recent anomalies
psql -U postgres -d bgp_monitoring -c "SELECT timestamp, peer, ensemble_score FROM ml_results WHERE ensemble_anomaly = true ORDER BY timestamp DESC LIMIT 10;"
```

---

**System Ready! 🚀 All services should now be running and dashboard should show live data.**
