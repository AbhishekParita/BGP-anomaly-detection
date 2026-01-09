# Database Schema - Phase 1 Complete ✅

## Overview
Complete PostgreSQL database schema for the BGP Anomaly Detection System with TimescaleDB for time-series optimization.

## Files Created

### 1. `schema_complete.sql`
**Purpose:** Complete database schema with all tables, views, functions, and indexes.

**What it includes:**
- **6 Main Tables:** Store BGP data, features, ML results, RPKI events, alerts, and metrics
- **2 Views:** Pre-built queries for dashboard (recent_alerts, alert_summary_hourly)
- **2 Functions:** Helper functions for scoring and alert management
- **Retention Policies:** Automatic data cleanup (30-365 days based on table)
- **Indexes:** Optimized for fast queries on time-series data

### 2. `init_database.py`
**Purpose:** Automated database initialization script.

**What it does:**
- Connects to PostgreSQL database
- Checks TimescaleDB extension availability
- Executes complete schema creation
- Verifies all tables, views, and functions were created
- Shows database size and next steps

### 3. `insert_sample_data.py`
**Purpose:** Insert realistic sample data for testing.

**What it does:**
- Generates 100 raw BGP records
- Creates 50 aggregated feature records
- Simulates 50 ML detection results
- Adds 20 RPKI validation events
- Creates 15 sample alerts (critical, high, medium, low)
- Adds 30 system performance metrics

---

## Database Tables Explained

### 📊 **Table 1: raw_bgp_data**
**Purpose:** Stores raw BGP updates directly from RIS Live WebSocket

**Key Fields:**
- `timestamp` - When the BGP update occurred
- `peer_addr` - IP address of BGP peer
- `prefix` - BGP prefix (e.g., 192.0.2.0/24)
- **9 Features:** announcements, withdrawals, total_updates, withdrawal_ratio, flap_count, path_length, unique_peers, message_rate, session_resets

**Data Flow:** RIS Live Collector → raw_bgp_data

**Why needed:** This is Layer 1 - captures all raw BGP events for processing

**Retention:** 30 days (older data auto-deleted to save space)

---

### 📊 **Table 2: features**
**Purpose:** Time-windowed aggregated features (1-minute windows)

**Key Fields:**
- `timestamp` - Window start time
- `window_duration` - Size of aggregation window (60 seconds = 1 minute)
- **9 Aggregated Features:** Sum/average of raw data over the time window
- **Statistical Features:** std_path_length, max_updates (for ML models)

**Data Flow:** Feature Aggregator reads raw_bgp_data → aggregates → writes to features

**Why needed:** ML models work on aggregated data, not individual events. This reduces noise and captures patterns over time.

**Retention:** 90 days

**Example:**
```
Window: 2026-01-07 10:00:00 to 10:01:00
- 50 announcements
- 10 withdrawals
- Average path length: 5.2
→ Creates 1 row in features table
```

---

### 📊 **Table 3: ml_results**
**Purpose:** Stores outputs from all three detectors (LSTM + IF + Heuristic)

**Key Fields:**
- **LSTM Scores:** reconstruction_error, anomaly_score, is_anomaly
- **Isolation Forest Scores:** anomaly_score, is_anomaly
- **Heuristic Scores:** score, reasons (array of triggered rules), is_anomaly
- **Ensemble:** combined score, confidence level

**Data Flow:** Detection Service (ensemble_bgp_optimized.py) → ml_results

**Why needed:** Keep individual detector scores for:
1. Debugging - see which detector triggered
2. Analysis - compare detector performance
3. Tuning - adjust weights based on accuracy

**Retention:** 60 days

---

### 📊 **Table 4: route_monitor_events**
**Purpose:** RPKI validation results and route security events

**Key Fields:**
- `rpki_status` - 'valid', 'invalid', or 'unknown'
- `event_type` - 'rpki_invalid', 'hijack_suspected', 'leak_suspected'
- `severity` - critical, high, medium, low
- `prefix` - Affected BGP prefix
- `origin_asn` - Origin AS number

**Data Flow:** RPKI Validator → route_monitor_events

**Why needed:** RPKI validation provides independent signal for:
- Route hijacking detection
- Route leak detection
- Adds security layer to ML detection

**Retention:** 180 days (longer for security compliance)

---

### 📊 **Table 5: alerts**
**Purpose:** Final correlated alerts shown in dashboard and sent to users

**Key Fields:**
- `alert_uuid` - Unique identifier for external systems
- `severity` - critical, high, medium, low
- `confidence` - How confident we are (0-1)
- `title` & `description` - Human-readable alert info
- `status` - 'open', 'acknowledged', 'resolved', 'false_positive'
- `ml_result_id` - Links back to ML detection
- `final_score` - Correlated score from all signals

**Data Flow:** Correlation Engine → alerts → API → Dashboard

**Why needed:** This is what users see! Single source of truth for:
- Dashboard display
- Email notifications
- ITSM ticket creation
- Alert lifecycle management

**Retention:** 365 days (1 year for historical analysis)

---

### 📊 **Table 6: system_metrics**
**Purpose:** Track system health and performance

**Key Fields:**
- `metric_name` - 'ingestion_rate', 'detection_latency', 'cpu_usage', etc.
- `metric_value` - Numeric value
- `component` - Which service (ris_collector, ml_detector, api, etc.)

**Data Flow:** All services → system_metrics

**Why needed:** Monitor system health, debug performance issues

**Retention:** 30 days

---

## Database Views

### 🔍 **View: recent_alerts**
Pre-joined query that combines:
- Alert data
- ML scores
- Heuristic reasons
- Count of route monitor events

**Use case:** Fast dashboard queries without complex JOINs

### 🔍 **View: alert_summary_hourly**
Hourly statistics:
- Alert count by severity
- Average confidence
- Average scores

**Use case:** Dashboard charts showing trends over time

---

## Database Functions

### ⚙️ **Function: calculate_ensemble_score()**
**Purpose:** Combine scores from 3 detectors with weights

**Parameters:**
- `lstm_score` (weight: 0.4)
- `if_score` (weight: 0.3)
- `heuristic_score` (weight: 0.3)

**Returns:** Weighted ensemble score (0-1)

**Use case:** Correlation Engine uses this to combine signals

### ⚙️ **Function: update_alert_status()**
**Purpose:** Update alert lifecycle status

**Parameters:**
- `alert_id` - Which alert
- `new_status` - 'acknowledged', 'resolved', etc.
- `user` - Who performed the action
- `notes` - Optional notes

**Use case:** API calls this when users acknowledge/resolve alerts

---

## Data Flow Architecture

```
Layer 1: Data Ingestion
RIS Live WebSocket → raw_bgp_data table

Layer 2: Feature Processing
Feature Aggregator reads raw_bgp_data
  → Aggregates 1-minute windows
  → Writes to features table

Layer 3: Detection
Detection Service reads features table
  → Runs LSTM + IF + Heuristic
  → Writes to ml_results table

RPKI Validator
  → Writes to route_monitor_events table

Correlation Engine reads:
  - ml_results
  - route_monitor_events
  → Combines signals
  → Writes to alerts table

Layer 4: Presentation
FastAPI reads alerts table
  → Serves to Dashboard
  → Sends notifications
```

---

## Usage Instructions

### Step 1: Initialize Database

```bash
# Make sure PostgreSQL is running
# Install TimescaleDB extension first

# Run initialization script
python database/init_database.py
```

**What happens:**
- Connects to PostgreSQL
- Checks TimescaleDB extension
- Creates all tables, views, functions
- Sets up indexes and retention policies
- Verifies everything was created correctly

### Step 2: Insert Sample Data (Optional - for testing)

```bash
python database/insert_sample_data.py
```

**What happens:**
- Inserts 100 raw BGP records
- Creates 50 feature records
- Generates 50 ML results
- Adds 20 RPKI events
- Creates 15 alerts
- Shows data summary

### Step 3: Verify Setup

```sql
-- Connect to database
psql -U postgres -d bgp_monitor

-- Check tables
\dt

-- View sample alerts
SELECT * FROM recent_alerts LIMIT 10;

-- Check alert statistics
SELECT * FROM alert_summary_hourly;
```

---

## Configuration

### Environment Variables (.env file)

```bash
# Database credentials
DB_NAME=bgp_monitor
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

### Retention Policies (Auto-cleanup)

| Table | Retention | Why |
|-------|-----------|-----|
| raw_bgp_data | 30 days | High volume, needed for recent analysis |
| features | 90 days | Processed data, longer for ML retraining |
| ml_results | 60 days | Detection results for tuning |
| route_monitor_events | 180 days | Security compliance |
| alerts | 365 days | Historical analysis, reporting |
| system_metrics | 30 days | Recent performance monitoring |

**How it works:** TimescaleDB automatically deletes data older than retention period

---

## Performance Optimizations

### 1. **TimescaleDB Hypertables**
All time-series tables are hypertables:
- Automatic partitioning by time (1-day chunks)
- Faster queries on time ranges
- Efficient compression

### 2. **Indexes**
Created indexes for common query patterns:
- `peer_addr + timestamp` - Query by specific peer
- `severity + timestamp` - Query by alert severity
- `status + severity` - Query open critical alerts

### 3. **Views**
Pre-computed JOINs reduce query complexity

---

## Next Steps (Phase 2)

Now that database is ready:

1. ✅ **Build RIS Live Collector** - Stream BGP data into raw_bgp_data
2. ✅ **Build Feature Aggregator** - Aggregate raw data into features table
3. ✅ **Integrate Detection Service** - Run ensemble_bgp_optimized.py continuously
4. ✅ **Build Correlation Engine** - Combine signals into final alerts
5. ✅ **Build FastAPI Backend** - REST API to query alerts
6. ✅ **Build Simple Dashboard** - Display alerts to users

---

## Troubleshooting

### Problem: "TimescaleDB extension not found"
**Solution:**
```bash
# Install TimescaleDB
# Ubuntu/Debian:
sudo apt install timescaledb-2-postgresql-14

# Enable in PostgreSQL
sudo -u postgres psql
CREATE EXTENSION timescaledb;
```

### Problem: "Permission denied"
**Solution:**
```bash
# Grant permissions
GRANT ALL PRIVILEGES ON DATABASE bgp_monitor TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
```

### Problem: "Connection refused"
**Solution:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start if needed
sudo systemctl start postgresql
```

---

## Summary

**Phase 1 Complete! ✅**

Created:
- ✅ Complete database schema (6 tables, 2 views, 2 functions)
- ✅ Automated initialization script
- ✅ Sample data insertion for testing
- ✅ Comprehensive documentation

**Ready for:** Phase 2 - Data Ingestion Pipeline
