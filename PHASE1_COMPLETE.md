# Phase 1: Database Schema - COMPLETE ✅

## 🎉 What We Just Built

### Files Created (in `database/` folder):

1. **`schema_complete.sql`** (400+ lines)
   - Complete PostgreSQL schema with TimescaleDB
   - 6 main tables + 1 helper table
   - 2 views for easy querying
   - 2 functions for business logic
   - Automatic retention policies
   - Performance-optimized indexes

2. **`init_database.py`** (Python script)
   - Automated database setup
   - Checks prerequisites (TimescaleDB)
   - Creates all schema objects
   - Verifies successful creation
   - User-friendly output with emoji indicators

3. **`insert_sample_data.py`** (Python script)
   - Generates realistic test data
   - 100+ records across all tables
   - Perfect for testing API and dashboard
   - Shows data summary after insertion

4. **`README.md`** (Documentation)
   - Complete documentation
   - Table explanations
   - Data flow diagrams
   - Usage instructions
   - Troubleshooting guide

---

## 📚 Key Concepts Explained

### Why 5 Separate Tables?

**Think of it like a factory production line:**

1. **raw_bgp_data** = Raw materials arriving
   - Individual BGP messages come in
   - Stored exactly as received
   - High volume, short retention (30 days)

2. **features** = Processed materials
   - Raw data aggregated into 1-minute windows
   - Noise removed, patterns emerge
   - This is what ML models consume

3. **ml_results** = Quality inspection reports
   - Each detector gives a score
   - Keep individual scores for analysis
   - See which detector triggered

4. **route_monitor_events** = Security inspection
   - RPKI validation results
   - Independent verification
   - Catches hijacks/leaks

5. **alerts** = Final products for customers
   - Correlated, high-confidence alerts
   - What users see in dashboard
   - What triggers notifications

### Why Not One Big Table?

**Performance & Clarity:**
- Separate concerns (raw data vs processed vs alerts)
- Query speed (smaller tables with focused indexes)
- Data lifecycle (different retention periods)
- Clear data flow (each stage has purpose)

---

## 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RIS Live WebSocket sends BGP UPDATE message             │
│    Example: "Peer 203.0.113.1 announced 50 routes"         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Collector extracts 9 features and writes to:            │
│    raw_bgp_data table                                       │
│    - timestamp: 2026-01-07 10:00:15                        │
│    - peer_addr: 203.0.113.1                                │
│    - announcements: 50                                      │
│    - withdrawals: 10                                        │
│    - path_length: 5.2                                       │
│    - ...                                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Feature Aggregator (runs every 1 minute)                │
│    - Reads last 60 seconds of raw_bgp_data                 │
│    - Aggregates: SUM announcements, AVG path_length, etc.  │
│    - Writes to: features table                             │
│    Result: 1 row representing 1-minute window              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Detection Service (ensemble_bgp_optimized.py)           │
│    - Reads from: features table                            │
│    - Runs 3 detectors:                                     │
│      * LSTM: reconstruction_error = 0.85 (HIGH!)          │
│      * Isolation Forest: score = -0.7 (ANOMALY!)          │
│      * Heuristic: score = 0.9, reason=CRITICAL_CHURN      │
│    - Calculates ensemble_score = 0.82                      │
│    - Writes to: ml_results table                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RPKI Validator (parallel process)                       │
│    - Validates prefix announcement                         │
│    - rpki_status = 'invalid' ⚠️                           │
│    - Writes to: route_monitor_events table                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Correlation Engine                                      │
│    - Reads: ml_results (score=0.82) + route_monitor (rpki) │
│    - Combines: ML says anomaly + RPKI says invalid         │
│    - Decision: HIGH CONFIDENCE ALERT                       │
│    - Writes to: alerts table                               │
│      * severity: 'critical'                                │
│      * confidence: 0.95                                    │
│      * title: "Route Hijack Suspected"                     │
│      * status: 'open'                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. FastAPI Backend                                         │
│    - Reads: alerts table (via SQL query)                   │
│    - Returns JSON to dashboard                             │
│    - Sends email notification                              │
│    - Creates ITSM ticket                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. User sees alert in Dashboard                            │
│    🚨 CRITICAL: Route Hijack Suspected                     │
│    Peer: 203.0.113.1                                       │
│    Confidence: 95%                                         │
│    [Acknowledge] [Mark as False Positive]                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Database Design Decisions

### 1. TimescaleDB Hypertables
**What:** Special PostgreSQL tables optimized for time-series data
**Why:** 
- Automatic partitioning by time (1-day chunks)
- 10-100x faster queries on time ranges
- Automatic data compression
- Built-in retention policies

**Example:**
```sql
-- Regular PostgreSQL: slow on large tables
SELECT * FROM raw_bgp_data 
WHERE timestamp > NOW() - INTERVAL '1 hour';
-- Scans millions of rows

-- TimescaleDB: blazing fast
-- Only looks at recent chunk (today's data)
-- Skips old chunks automatically
```

### 2. Retention Policies
**What:** Automatic data deletion after X days
**Why:**
- Save storage space
- Comply with data retention policies
- Keep database performant

**How it works:**
```sql
SELECT add_retention_policy('raw_bgp_data', INTERVAL '30 days');
-- Data older than 30 days is automatically deleted
-- Runs in background, no manual intervention
```

### 3. Indexes
**What:** Special data structures for fast lookups
**Why:** 
- Without index: scan entire table (slow)
- With index: jump directly to relevant rows (fast)

**Example:**
```sql
-- Query: Find all alerts for specific peer
SELECT * FROM alerts WHERE peer_addr = '203.0.113.1';

-- Without index: Scans all rows
-- With idx_alerts_peer: Instant lookup
```

### 4. Views
**What:** Pre-defined queries saved as virtual tables
**Why:**
- Simplify complex queries
- Consistent query logic
- Easier for API developers

**Example:**
```sql
-- Instead of complex JOIN every time:
SELECT a.*, m.ensemble_score, COUNT(r.id) 
FROM alerts a 
LEFT JOIN ml_results m ON a.ml_result_id = m.id
LEFT JOIN route_monitor_events r ...
-- (50 lines of SQL)

-- Just use the view:
SELECT * FROM recent_alerts;
-- (1 line, same result)
```

### 5. Functions
**What:** Reusable SQL logic
**Why:**
- Consistent calculations
- Encapsulate business logic
- Called from application code

**Example:**
```sql
-- Calculate weighted ensemble score
SELECT calculate_ensemble_score(0.8, -0.6, 0.9);
-- Returns: 0.76 (weighted combination)
```

---

## 🚀 How to Use (Step by Step)

### Prerequisites
```bash
# 1. PostgreSQL 14+ installed and running
sudo systemctl status postgresql

# 2. Python virtual environment activated
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate      # Linux/Mac

# 3. Install Python dependencies
pip install psycopg2-binary python-dotenv
```

### Step 1: Create Database
```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE bgp_monitor;

# Install TimescaleDB extension
\c bgp_monitor
CREATE EXTENSION timescaledb;

# Exit
\q
```

### Step 2: Update .env File
```bash
# Edit .env file in project root
DB_NAME=bgp_monitor
DB_USER=postgres
DB_PASSWORD=your_password  # Change this!
DB_HOST=localhost
DB_PORT=5432
```

### Step 3: Initialize Schema
```bash
# Run initialization script
python database/init_database.py

# Expected output:
# ========================================
# BGP ANOMALY DETECTION - DATABASE INITIALIZATION
# ========================================
# 🔌 Connecting to PostgreSQL database...
#    Host: localhost:5432
#    Database: bgp_monitor
# ✅ Connected successfully!
# 
# 🔍 Checking TimescaleDB extension...
# ✅ TimescaleDB extension is enabled
# 
# 📄 Loading schema from: database/schema_complete.sql
# ⚙️  Executing schema creation...
# ✅ Schema executed successfully!
# 
# ✓ Verifying table creation...
#    ✅ raw_bgp_data: Created (rows: 0)
#    ✅ features: Created (rows: 0)
#    ✅ ml_results: Created (rows: 0)
#    ✅ route_monitor_events: Created (rows: 0)
#    ✅ alerts: Created (rows: 0)
#    ✅ system_metrics: Created (rows: 0)
# 
# ✓ Verifying views...
#    ✅ recent_alerts
#    ✅ alert_summary_hourly
# 
# ========================================
# ✅ DATABASE INITIALIZATION COMPLETE!
# ========================================
```

### Step 4: Insert Sample Data (Optional)
```bash
# Insert test data
python database/insert_sample_data.py

# Expected output:
# ========================================
# INSERTING SAMPLE DATA
# ========================================
# 🔌 Connecting to database...
# ✅ Connected!
# 
# 📊 Inserting 100 raw BGP records...
# ✅ Inserted 100 raw BGP records
# 
# 📊 Inserting 50 feature records...
# ✅ Inserted 50 feature records
# 
# ... (continues for all tables)
# 
# ========================================
# 📊 DATA SUMMARY
# ========================================
#    raw_bgp_data: 100 records
#    features: 50 records
#    ml_results: 50 records
#    route_monitor_events: 20 records
#    alerts: 15 records
#    system_metrics: 30 records
# ========================================
```

### Step 5: Verify with SQL
```bash
# Connect to database
psql -U postgres -d bgp_monitor

# View sample alerts
SELECT id, severity, title, peer_addr, confidence 
FROM alerts 
ORDER BY timestamp DESC 
LIMIT 5;

# Check alert statistics
SELECT * FROM alert_summary_hourly LIMIT 10;

# Count records in each table
SELECT 
    'raw_bgp_data' as table_name, COUNT(*) as rows FROM raw_bgp_data
UNION ALL
SELECT 'features', COUNT(*) FROM features
UNION ALL
SELECT 'ml_results', COUNT(*) FROM ml_results
UNION ALL
SELECT 'alerts', COUNT(*) FROM alerts;
```

---

## 📊 Sample Queries for Testing

### Query 1: Get Critical Alerts
```sql
SELECT 
    timestamp,
    severity,
    title,
    peer_addr,
    confidence,
    status
FROM alerts
WHERE severity = 'critical'
    AND status = 'open'
ORDER BY timestamp DESC;
```

### Query 2: Peer Anomaly Frequency
```sql
SELECT 
    peer_addr,
    COUNT(*) as alert_count,
    AVG(confidence) as avg_confidence,
    MAX(final_score) as max_score
FROM alerts
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY peer_addr
ORDER BY alert_count DESC;
```

### Query 3: Detection Method Comparison
```sql
SELECT 
    CASE 
        WHEN ensemble_score > 0.8 THEN 'High Confidence'
        WHEN ensemble_score > 0.5 THEN 'Medium Confidence'
        ELSE 'Low Confidence'
    END as confidence_level,
    COUNT(*) as alert_count,
    AVG(ensemble_score) as avg_score
FROM alerts
GROUP BY confidence_level;
```

### Query 4: RPKI Validation Status
```sql
SELECT 
    rpki_status,
    event_type,
    COUNT(*) as event_count
FROM route_monitor_events
GROUP BY rpki_status, event_type
ORDER BY event_count DESC;
```

### Query 5: System Performance
```sql
SELECT 
    component,
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    unit
FROM system_metrics
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY component, metric_name, unit
ORDER BY component, metric_name;
```

---

## ✅ Phase 1 Completion Checklist

- [x] Complete SQL schema created (schema_complete.sql)
- [x] 6 main tables with proper structure
- [x] TimescaleDB hypertables configured
- [x] Indexes for performance optimization
- [x] Views for simplified querying
- [x] Helper functions for business logic
- [x] Retention policies for data lifecycle
- [x] Automated initialization script (init_database.py)
- [x] Sample data generation script (insert_sample_data.py)
- [x] Complete documentation (README.md)
- [x] .env file configuration

---

## 🎯 What's Next: Phase 2 Preview

Now that database is ready, we'll build:

### Phase 2: Data Ingestion Pipeline
1. **RIS Live Collector Service**
   - WebSocket connection to RIS Live
   - Parse BGP messages
   - Extract 9 features
   - Insert into raw_bgp_data

2. **Feature Aggregator Service**
   - Read raw_bgp_data every minute
   - Aggregate features over time window
   - Write to features table

**Expected time:** 2-3 days

---

## 🎓 Learning Summary

**You now understand:**
1. ✅ Why we need separate tables (separation of concerns)
2. ✅ How TimescaleDB optimizes time-series data
3. ✅ What each table stores and why
4. ✅ How data flows through the system (8 steps)
5. ✅ Why retention policies save space
6. ✅ How indexes speed up queries
7. ✅ What views and functions do

**Database is production-ready for:**
- High-volume BGP data ingestion
- Real-time anomaly detection
- Fast dashboard queries
- Long-term data retention with automatic cleanup

---

## 🆘 Common Issues & Solutions

### Issue: "database bgp_monitor does not exist"
```bash
# Solution: Create database first
psql -U postgres -c "CREATE DATABASE bgp_monitor;"
```

### Issue: "extension timescaledb not found"
```bash
# Solution: Install TimescaleDB
# Ubuntu/Debian:
sudo apt-get install timescaledb-2-postgresql-14

# Then enable:
psql -U postgres -d bgp_monitor -c "CREATE EXTENSION timescaledb;"
```

### Issue: "permission denied"
```bash
# Solution: Grant permissions
psql -U postgres -d bgp_monitor -c "GRANT ALL PRIVILEGES ON DATABASE bgp_monitor TO postgres;"
```

### Issue: "connection refused on port 5432"
```bash
# Solution: Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Auto-start on boot
```

---

**Phase 1 Complete! 🎉**

Ready to move to Phase 2? Type "start phase 2" when ready!
