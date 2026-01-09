# 🔍 Model Drift Detection - Complete Technical Analysis

## Overview
The drift monitoring system continuously analyzes model performance by comparing baseline metrics (7 days old) with current metrics (last 24 hours) to detect when models degrade or shift behavior.

---

## 📊 Drift Detection Algorithms & Parameters

### 1. **Score Distribution Shift (Primary Method)**

**Algorithm**: Normalized Mean Shift (Z-score approximation)

```
score_shift = |current_mean - baseline_mean| / (baseline_std + ε)
```

**Parameters**:
- `baseline_window`: 7 days (168 hours) of historical data
- `current_window`: 24 hours of recent data
- Threshold for LSTM: **0.15** (15% shift)
- Threshold for Isolation Forest: **0.15** (15% shift)
- ε (epsilon): 1e-6 (prevents division by zero)

**What it detects**:
- When average anomaly scores shift significantly from baseline
- Example: If baseline mean = 0.45, current mean = 0.65 → shift = 0.2 (20%) → **DRIFT DETECTED**
- Works well for detecting gradual model performance decay

**Why it works**:
- Normalized by standard deviation (accounts for data variability)
- Sensitive to both positive and negative shifts
- Single number metric for easy monitoring

---

### 2. **Anomaly Rate Change (Secondary Method)**

**Algorithm**: Relative Rate Difference

```
anomaly_rate_change = |current_rate - baseline_rate| / (baseline_rate + ε)
```

**Parameters**:
- Current anomaly rate: proportion of detections in last 24 hours
- Baseline anomaly rate: proportion of detections in 7-day window
- Threshold for LSTM: **0.20** (20% change)
- Threshold for Isolation Forest: **0.20** (20% change)
- Threshold for Heuristic: **0.30** (30% change - higher due to rule-based nature)

**What it detects**:
- Sudden changes in detection frequency
- Example: If baseline rate = 5%, current rate = 8% → change = 60% → **DRIFT DETECTED**
- Indicates model is flagging more/fewer anomalies than expected

**Why it matters**:
- Catches changes in decision boundaries
- Detects if model becomes overly sensitive or insensitive
- Different threshold for heuristic (rules are more stable)

---

### 3. **Distribution Quartile Shift (Tertiary Method)**

**Algorithm**: Mean Quartile Deviation

```
q_shift = mean(|Q25_current - Q25_baseline|, 
               |Q50_current - Q50_baseline|, 
               |Q75_current - Q75_baseline|)
```

**Quantiles Used**:
- Q25 (25th percentile) - Lower quartile
- Q50 (50th percentile) - Median
- Q75 (75th percentile) - Upper quartile
- Q90 (90th percentile) - Tracked but not in main shift calculation

**Threshold**: **0.15** (15% quartile shift)

**What it detects**:
- Distribution shape changes, not just mean shifts
- When score distribution becomes skewed or bimodal
- Example: If Q50 shifts from 0.4 to 0.6 → **DRIFT DETECTED**

**Why it works**:
- Captures distribution character changes
- Q25/Q50/Q75 shift together if data transforms uniformly
- Catches cases where mean is stable but distribution changes

---

## 🔧 Data Collection & Metrics Computed

### Baseline Metrics (Calculated from 7-day historical data)

| Metric | Purpose | Calculation |
|--------|---------|-------------|
| `mean_score` | Average anomaly score | `np.mean(scores)` |
| `std_score` | Standard deviation | `np.std(scores)` - measures variability |
| `median_score` | Middle value | `np.percentile(scores, 50)` |
| `anomaly_rate` | % of anomalies detected | `np.mean(is_anomaly)` |
| `q25, q50, q75, q90` | Distribution quartiles | `np.percentile(scores, [25,50,75,90])` |
| `sample_count` | Data points used | Count of non-null scores |
| `timestamp` | When baseline was calculated | ISO format datetime |

### Current Metrics (Calculated from 24-hour recent data)

Same metrics as baseline but from recent time window for comparison.

---

## ⚙️ Processing Flow

```
Every 1 Hour:
    ├─ Calculate baseline metrics (7 days old → 1 day old)
    ├─ Calculate current metrics (last 24 hours)
    │
    ├─ For each model (LSTM, IF, Heuristic):
    │   ├─ Calculate score_shift
    │   ├─ Calculate anomaly_rate_change
    │   ├─ Calculate quartile_shift
    │   │
    │   ├─ Check if ANY metric exceeds threshold
    │   │   └─ If YES → Drift Detected
    │   │
    │   └─ If Drift Detected:
    │       ├─ Save drift report to JSON
    │       ├─ Create retraining flag file
    │       ├─ Trigger retraining script (background process)
    │       └─ Log event to database
    │
    └─ Wait 1 hour, repeat
```

---

## 🎯 Thresholds Summary

### LSTM Detector
| Metric | Threshold | Impact |
|--------|-----------|--------|
| Score Shift | 0.15 | Detects 15% mean score change |
| Anomaly Rate Change | 0.20 | Detects 20% rate change |
| Quartile Shift | 0.15 | Detects 15% distribution shift |

### Isolation Forest Detector
| Metric | Threshold | Impact |
|--------|-----------|--------|
| Score Shift | 0.15 | Detects 15% mean score change |
| Anomaly Rate Change | 0.20 | Detects 20% rate change |
| Quartile Shift | 0.15 | Detects 15% distribution shift |

### Heuristic Detector (Rule-based)
| Metric | Threshold | Impact |
|--------|-----------|--------|
| Anomaly Rate Change | 0.30 | Higher threshold (rules are stable) |
| Quartile Shift | 0.15 | Rules produce discrete scores |

---

## 📈 Real-World Example

**Scenario**: LSTM model drift detection over time

### Hour 0 (Baseline Established)
```
Baseline metrics from 7-168 hours ago:
├─ mean_score = 0.45
├─ std_score = 0.20
├─ median_score = 0.42
├─ anomaly_rate = 5.2%
└─ q50 (median) = 0.42
```

### Hour 168 (24 hours of current data analyzed)
```
Current metrics (last 24 hours):
├─ mean_score = 0.58      ← SHIFTED UP!
├─ std_score = 0.18
├─ median_score = 0.55
├─ anomaly_rate = 9.1%    ← INCREASED!
└─ q50 (median) = 0.55

Drift Calculations:
├─ score_shift = |0.58 - 0.45| / (0.20 + ε) = 0.65 ✓ EXCEEDS 0.15
├─ anomaly_rate_change = |0.091 - 0.052| / (0.052 + ε) = 0.75 ✓ EXCEEDS 0.20
└─ quartile_shift = mean of [|0.55-0.42|, ...] = 0.13 ✗ Below 0.15

Result: ⚠️ DRIFT DETECTED
├─ Reason 1: Score shift (0.65 > 0.15)
├─ Reason 2: Anomaly rate change (0.75 > 0.20)
└─ Action: Trigger LSTM retraining
```

---

## 🚀 Automatic Retraining Workflow

When drift is detected:

1. **Log Event**
   - Save drift report to JSON file with all metrics
   - Record in database `drift_flags` table
   
2. **Create Flag File**
   - Write `retrain_[model].flag` with drift metrics
   - Acts as signal for retraining script
   
3. **Trigger Retraining**
   - Launch `retrain_[model].py` in background process
   - Detection continues with old model (no service interruption)
   
4. **Hot-Swap**
   - When retraining completes, new model replaces old one
   - Zero downtime model updates
   
5. **Update Baseline**
   - Baseline recalculated on next drift check cycle
   - Prevents cascading false positives

---

## 📊 Monitoring Metrics

The system tracks:

```python
Statistics tracked:
├─ checks_performed: Total drift checks run
├─ drifts_detected: Number of times drift was detected
├─ retraining_triggered: Times retraining was automatically started
└─ last_check: Timestamp of most recent check
```

**Dashboard Display**:
- Shows "Healthy" (green) when no recent drift
- Shows "Drift Detected" (yellow) when drift flag is active
- Shows retraining status for each model

---

## 🔐 Advantages of This Approach

✅ **Multi-method Detection**
   - Uses 3 independent methods (score shift, rate change, quartile shift)
   - Reduces false positives (must exceed threshold)
   
✅ **Handles Different Model Types**
   - LSTM: Continuous scores (deep learning)
   - Isolation Forest: Continuous scores (statistical)
   - Heuristic: Discrete rules (threshold-based)
   - Separate thresholds for each
   
✅ **Adaptive Baselines**
   - 7-day baseline accounts for weekly patterns
   - 24-hour current window catches rapid changes
   - Automatic recalculation after retraining
   
✅ **Automatic Recovery**
   - No manual intervention required
   - Retraining triggered automatically
   - Detection continues uninterrupted
   
✅ **Production-Ready**
   - Normalized metrics prevent scale sensitivity
   - Epsilon prevents division by zero
   - Error handling for missing data
   - Continuous monitoring with logging

---

## ⚠️ Limitations & Future Improvements

**Current Limitations**:
1. No ground truth data (can't measure real accuracy)
2. Assumes 7-day baseline is representative
3. Thresholds are fixed (could be adaptive)
4. No temporal patterns (e.g., weekend vs weekday)

**Possible Improvements**:
1. Add ground truth feedback when anomalies are confirmed/denied
2. Implement adaptive thresholds based on recent drift history
3. Use CUSUM (Cumulative Sum Control Chart) for sequential drift
4. Add seasonal decomposition for time-series patterns
5. Implement ADWIN algorithm for concept drift detection

---

## 📝 Configuration Files

Located in `services/drift_monitor.py`:

```python
# Time windows
self.check_interval = 3600          # Every hour
self.baseline_window = 7 * 24       # 7 days (168 hours)
self.current_window = 24            # Last 24 hours

# Thresholds (adjustable)
self.drift_thresholds = {
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
```

---

## 🎓 Key Takeaways

| Component | Purpose | Formula |
|-----------|---------|---------|
| **Score Shift** | Detects mean score changes | `\|μ_current - μ_baseline\| / σ_baseline` |
| **Rate Change** | Detects anomaly frequency changes | `\|rate_current - rate_baseline\| / rate_baseline` |
| **Quartile Shift** | Detects distribution shape changes | `mean(\|Q_current - Q_baseline\|)` |
| **Thresholds** | Trigger sensitivity | Model-specific values (0.15-0.30) |
| **Baseline Window** | Historical reference period | 7 days (168 hours) |
| **Current Window** | Comparison period | 24 hours |
| **Check Interval** | How often to monitor | 1 hour (3600 seconds) |

---

**System actively monitoring all 3 models with automatic retraining on drift detection! 🔄**
