# Model Drift Detection - Quick Start Guide

## What is Model Drift?

Model drift happens when ML models become less accurate over time because:
- Network traffic patterns change
- New attack types emerge
- Normal behavior evolves

**Solution:** Automatically detect drift and retrain models to maintain accuracy.

---

## Quick Start (5 minutes)

### Step 1: Test the retraining system

```bash
python test_retraining.py
```

This will:
- Test Isolation Forest retraining (~30 seconds)
- Test Heuristic rules update (~10 seconds)
- Ask if you want to test LSTM (optional, takes ~5 minutes)
- Verify hot-swap mechanism works

**Expected output:**
```
============================================================
TEST SUMMARY
============================================================
Isolation Forest     ✅ PASSED
Heuristic            ✅ PASSED
LSTM                 ⏭️ SKIPPED (optional)
Hot-Swap             ✅ PASSED
============================================================
Result: 3/3 tests passed
```

---

### Step 2: Start drift monitoring

```bash
python services/drift_monitor.py
```

This starts the drift monitor service which:
- Checks all 3 models every hour
- Detects performance degradation
- Automatically triggers retraining when drift detected
- Runs continuously in background

**You'll see:**
```
============================================================
MODEL DRIFT MONITOR SERVICE STARTED
============================================================
Check interval: 3600 seconds (1.0 hours)
Baseline window: 168 hours (7.0 days)
Current window: 24 hours
Monitoring models: LSTM, Isolation Forest, Heuristic
============================================================
```

---

### Step 3: Check drift status

Open a new terminal and run:

```bash
python check_drift_status.py
```

This shows:
- Current drift detection status
- Latest drift reports
- Model versions and ages
- Retraining history

**Example output:**
```
============================================================
DRIFT MONITOR DASHBOARD
============================================================
✅ Service active (last activity: 15.2 minutes ago)

No active retraining flags
Latest drift reports: All OK
Models up to date
```

---

## What Happens Automatically?

### Normal Operation (No Drift)

```
Hour 0  ─→  Hour 1  ─→  Hour 2  ─→  Hour 3
   ✅          ✅          ✅          ✅
No drift   No drift   No drift   No drift
```

### When Drift Detected

```
Hour 12: Drift detected in Isolation Forest
    ↓
Automatic retraining starts (30 seconds)
    ↓
Old model backed up
    ↓
New model activated
    ↓
Detector reloads new model (10 seconds)
    ↓
Detection continues with improved model
```

**Key Point:** No system downtime! Detection continues during retraining.

---

## Manual Retraining

You can manually retrain any model:

### Retrain Isolation Forest (fast)
```bash
python retrain_isolation_forest.py
```

### Update Heuristic Rules (very fast)
```bash
python retrain_heuristic.py
```

### Retrain LSTM (slow)
```bash
python retrain_lstm.py
```

Each script will:
1. Extract recent data (last 7 days)
2. Train new model
3. Validate performance
4. Hot-swap with current model
5. Backup old model

---

## Integration with Existing System

### Start ALL services (including drift monitor)

```bash
python run_all_services.py start
```

This starts 8 services:
1. RIS Live Collector
2. Feature Aggregator
3. Heuristic Detector
4. LSTM Detector
5. Isolation Forest Detector
6. Ensemble Coordinator
7. Correlation Engine
8. **Drift Monitor** (NEW)

### Check all service status

```bash
python run_all_services.py status
```

---

## Monitoring Schedule

**Recommended:**
- Check drift status: **Daily** (`python check_drift_status.py`)
- Review drift reports: **Weekly** (when drift detected)
- Test retraining: **Monthly** (`python test_retraining.py`)

---

## Understanding Drift Reports

When drift is detected, a report is saved: `model_artifacts/drift_report_<model>_<timestamp>.json`

**Example:**
```json
{
  "drift_detected": true,
  "model_name": "isolation_forest",
  "reasons": ["score_shift", "anomaly_rate_change"],
  "metrics": {
    "score_shift": 0.187,
    "anomaly_rate_change": 0.254
  },
  "timestamp": "2026-01-08T15:30:00"
}
```

**What it means:**
- `score_shift: 0.187` → Model scores changed by 18.7% (threshold: 15%)
- `anomaly_rate_change: 0.254` → Anomaly detection rate changed by 25.4% (threshold: 20%)
- **Action:** Model automatically retrained

---

## How Models Retrain Independently

```
┌────────────┐
│ IF Detector│ ← IF retraining (30s)
├────────────┤
│LSTM Detector│ ← Continue detecting
├────────────┤
│Heur Detector│ ← Continue detecting
└────────────┘
       ↓
  Ensemble
  continues
  working
```

**Benefits:**
- No system downtime
- Other models compensate during retraining
- Ensemble voting ensures reliability
- Hot-swap is seamless

---

## Troubleshooting

### "No baseline data" error

**Cause:** System needs 7 days of historical data

**Solution:** Wait for data to accumulate, or reduce training_days:
```python
# In retrain_*.py
self.training_days = 3  # Use 3 days instead of 7
```

### "Insufficient data for retraining"

**Cause:** Not enough recent samples

**Solution:** 
1. Check RIS collector is running: `python check_realtime.py`
2. Verify database has data: `python check_database.py`

### Drift monitor stopped

**Check:**
```bash
python check_drift_status.py
```

If stopped, restart:
```bash
python services/drift_monitor.py
```

Or restart all services:
```bash
python run_all_services.py restart
```

---

## Configuration

### Change check interval (how often to check for drift)

Edit `services/drift_monitor.py`:
```python
self.check_interval = 3600  # 1 hour (default)
# self.check_interval = 7200  # 2 hours
# self.check_interval = 1800  # 30 minutes
```

### Change drift thresholds (sensitivity)

Edit `services/drift_monitor.py`:
```python
self.drift_thresholds = {
    'lstm': {
        'score_shift_threshold': 0.15,      # 15% (default)
        'anomaly_rate_threshold': 0.20,     # 20% (default)
    },
    # Lower = more sensitive (retrain more often)
    # Higher = less sensitive (retrain less often)
}
```

### Change training data window

Edit retraining scripts:
```python
self.training_days = 7  # Use last 7 days (default)
# self.training_days = 14  # Use last 14 days
# self.training_days = 3   # Use last 3 days
```

---

## Performance

| Task | Time | Impact |
|------|------|--------|
| IF retraining | 30 seconds | None (hot-swap) |
| Heuristic update | 10 seconds | None (hot-swap) |
| LSTM retraining | 5-10 minutes | None (hot-swap) |
| Drift check | < 5 seconds | None (background) |

**System remains online during all operations.**

---

## Best Practices

✅ **DO:**
- Run drift monitor continuously
- Check status daily
- Review drift reports when created
- Test retraining monthly
- Keep backups (automatic)

❌ **DON'T:**
- Stop drift monitor in production
- Ignore drift warnings
- Delete backup files manually
- Change model files during detection
- Set check_interval < 30 minutes (too frequent)

---

## Production Deployment

### Option 1: Run in background (simple)

```bash
nohup python services/drift_monitor.py > drift_monitor.out 2>&1 &
```

### Option 2: Use process manager (recommended)

**Using PM2:**
```bash
pm2 start services/drift_monitor.py --name drift-monitor
pm2 save
pm2 startup
```

**Using Supervisor:**
```ini
[program:drift-monitor]
command=python services/drift_monitor.py
directory=/path/to/lstm_model
autostart=true
autorestart=true
```

---

## Summary

**3 Simple Steps:**

1. **Test:** `python test_retraining.py`
2. **Start:** `python services/drift_monitor.py`
3. **Monitor:** `python check_drift_status.py`

**That's it!** The system now automatically maintains model accuracy.

---

## Need Help?

**Check system status:**
```bash
python check_drift_status.py
```

**Test retraining:**
```bash
python test_retraining.py
```

**View logs:**
```bash
tail -f drift_monitor.log
```

**Manual retrain:**
```bash
python retrain_<model_name>.py
```

---

**Quick Start Complete!** 🎉

For detailed documentation, see [DRIFT_DETECTION_README.md](DRIFT_DETECTION_README.md)
