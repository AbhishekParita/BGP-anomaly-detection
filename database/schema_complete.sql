-- ============================================================================
-- BGP ANOMALY DETECTION SYSTEM - COMPLETE DATABASE SCHEMA
-- ============================================================================
-- Purpose: Complete database schema for the BGP anomaly detection pipeline
-- Created: January 7, 2026
-- Database: PostgreSQL 14+ with TimescaleDB extension
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- LAYER 2: STORAGE AND FEATURE PROCESSING TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table 1: raw_bgp_data
-- Purpose: Stores raw BGP updates with extracted 9 features from RIS Live
-- Data Flow: RIS Live Collector → raw_bgp_data
-- Retention: 30 days (configurable with TimescaleDB retention policy)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_bgp_data (
    id BIGSERIAL,                              -- Auto-incrementing primary key
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,  -- When BGP update occurred
    peer_addr TEXT NOT NULL,                   -- BGP peer IP address
    peer_asn BIGINT,                          -- Peer AS number
    prefix CIDR,                              -- BGP prefix (e.g., 192.0.2.0/24)
    
    -- 9 Core Features for ML Detection
    announcements INTEGER DEFAULT 0,           -- Number of route announcements
    withdrawals INTEGER DEFAULT 0,             -- Number of route withdrawals
    total_updates INTEGER DEFAULT 0,           -- Total updates (announcements + withdrawals)
    withdrawal_ratio REAL DEFAULT 0.0,         -- Ratio of withdrawals to announcements
    flap_count INTEGER DEFAULT 0,              -- Route flapping incidents
    path_length REAL DEFAULT 0.0,              -- Average AS path length
    unique_peers INTEGER DEFAULT 0,            -- Number of unique peers
    message_rate REAL DEFAULT 0.0,             -- BGP messages per second
    session_resets INTEGER DEFAULT 0,          -- BGP session reset count
    
    -- Metadata
    raw_message JSONB,                         -- Full BGP message in JSON format
    created_at TIMESTAMP DEFAULT NOW()         -- Record insertion time
);

-- Create TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('raw_bgp_data', 'timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'    -- Partition data by day
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_raw_bgp_peer ON raw_bgp_data(peer_addr, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_bgp_prefix ON raw_bgp_data(prefix, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_bgp_asn ON raw_bgp_data(peer_asn, timestamp DESC);

COMMENT ON TABLE raw_bgp_data IS 'Raw BGP updates from RIS Live with extracted features';

-- ----------------------------------------------------------------------------
-- Table 2: features
-- Purpose: Aggregated features over time windows (1-min, 5-min intervals)
-- Data Flow: Feature Aggregator reads raw_bgp_data → writes to features
-- Why needed: ML models work on aggregated time-series data, not raw events
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,  -- Window start time
    peer_addr TEXT NOT NULL,                   -- Peer being monitored
    peer_asn BIGINT,
    window_duration INTEGER DEFAULT 60,        -- Window size in seconds (60 = 1 min)
    
    -- Aggregated 9 Features (computed over the time window)
    announcements INTEGER DEFAULT 0,           -- Sum of announcements in window
    withdrawals INTEGER DEFAULT 0,             -- Sum of withdrawals in window
    total_updates INTEGER DEFAULT 0,           -- Total updates in window
    withdrawal_ratio REAL DEFAULT 0.0,         -- Avg withdrawal ratio
    flap_count INTEGER DEFAULT 0,              -- Total flaps in window
    path_length REAL DEFAULT 0.0,              -- Average path length
    unique_peers INTEGER DEFAULT 0,            -- Count of unique peers
    message_rate REAL DEFAULT 0.0,             -- Messages per second
    session_resets INTEGER DEFAULT 0,          -- Total resets in window
    
    -- Statistical features for ML
    std_path_length REAL DEFAULT 0.0,          -- Standard deviation of path length
    max_updates INTEGER DEFAULT 0,             -- Max updates in window
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable('features', 'timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_features_peer ON features(peer_addr, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_window ON features(window_duration, timestamp DESC);

COMMENT ON TABLE features IS 'Time-windowed aggregated features for ML model input';

-- ----------------------------------------------------------------------------
-- Table 3: ml_results
-- Purpose: Stores raw outputs from all three detectors (LSTM, IF, Heuristic)
-- Data Flow: Detection Service (ensemble_bgp_optimized.py) → ml_results
-- Why needed: Keep individual detector scores for analysis and debugging
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_results (
    id BIGSERIAL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    peer_addr TEXT NOT NULL,
    feature_id BIGINT,                         -- Links to features table
    
    -- Individual Detector Scores
    lstm_reconstruction_error REAL,            -- LSTM autoencoder error (0-1+)
    lstm_anomaly_score REAL,                   -- Normalized LSTM score (0-1)
    lstm_is_anomaly BOOLEAN DEFAULT FALSE,     -- LSTM binary classification
    
    if_anomaly_score REAL,                     -- Isolation Forest score (-1 to 1)
    if_is_anomaly BOOLEAN DEFAULT FALSE,       -- IF binary classification
    
    heuristic_score REAL,                      -- Rule-based score (0-1)
    heuristic_reasons TEXT[],                  -- Array of triggered rules
    heuristic_is_anomaly BOOLEAN DEFAULT FALSE,
    
    -- Ensemble Results
    ensemble_score REAL NOT NULL,              -- Weighted combined score (0-1)
    ensemble_confidence REAL,                  -- Confidence level (0-1)
    
    -- Processing metadata
    model_version TEXT,                        -- Model version used
    processing_time_ms REAL,                   -- Processing duration
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable('ml_results', 'timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ml_results_peer ON ml_results(peer_addr, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_results_anomaly ON ml_results(ensemble_score DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_results_feature ON ml_results(feature_id);

COMMENT ON TABLE ml_results IS 'Individual and ensemble ML detection results';

-- ----------------------------------------------------------------------------
-- Table 4: route_monitor_events
-- Purpose: RPKI validation results and route monitoring events
-- Data Flow: RPKI Validator/Route Monitor → route_monitor_events
-- Why needed: Provides additional validation signal for correlation
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS route_monitor_events (
    id BIGSERIAL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    peer_addr TEXT NOT NULL,
    peer_asn BIGINT,
    prefix CIDR NOT NULL,
    origin_asn BIGINT,
    
    -- RPKI Validation
    rpki_status TEXT,                          -- 'valid', 'invalid', 'unknown'
    roa_prefix CIDR,                          -- ROA matched prefix
    roa_max_length INTEGER,                    -- ROA max prefix length
    
    -- Route Monitoring Checks
    event_type TEXT NOT NULL,                  -- 'rpki_invalid', 'hijack_suspected', 'leak_suspected'
    severity TEXT,                             -- 'critical', 'high', 'medium', 'low'
    description TEXT,                          -- Human-readable description
    
    -- Additional context
    as_path TEXT,                              -- Full AS path
    next_hop INET,                            -- Next hop IP
    
    metadata JSONB,                            -- Additional event data
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable('route_monitor_events', 'timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_route_monitor_peer ON route_monitor_events(peer_addr, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_route_monitor_prefix ON route_monitor_events(prefix, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_route_monitor_rpki ON route_monitor_events(rpki_status, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_route_monitor_event_type ON route_monitor_events(event_type, severity);

COMMENT ON TABLE route_monitor_events IS 'RPKI validation and route monitoring events';

-- ----------------------------------------------------------------------------
-- Table 5: alerts
-- Purpose: Final correlated alerts for presentation layer (API/Dashboard)
-- Data Flow: Correlation Engine → alerts → FastAPI → Dashboard
-- Why needed: Single source of truth for alerts shown to users
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL,
    alert_uuid UUID DEFAULT gen_random_uuid(),  -- Unique alert identifier
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    
    -- Alert Classification
    alert_type TEXT NOT NULL,                  -- 'ml_anomaly', 'rpki_invalid', 'hybrid'
    severity TEXT NOT NULL,                    -- 'critical', 'high', 'medium', 'low'
    confidence REAL NOT NULL,                  -- Overall confidence (0-1)
    
    -- Affected Resources
    peer_addr TEXT NOT NULL,
    peer_asn BIGINT,
    affected_prefixes CIDR[],                  -- Array of affected prefixes
    
    -- Alert Details
    title TEXT NOT NULL,                       -- Short alert title
    description TEXT,                          -- Detailed description
    anomaly_types TEXT[],                      -- ['churn', 'path_length', 'rpki_invalid']
    
    -- Correlation Data (links to source tables)
    ml_result_id BIGINT,                       -- Links to ml_results
    route_monitor_event_ids BIGINT[],          -- Links to route_monitor_events
    
    -- Scoring Details
    ensemble_score REAL,                       -- From ML
    heuristic_score REAL,                      -- From rules
    rpki_score REAL,                          -- From RPKI validation
    final_score REAL NOT NULL,                 -- Correlated final score
    
    -- Alert Lifecycle
    status TEXT DEFAULT 'open',                -- 'open', 'acknowledged', 'resolved', 'false_positive'
    acknowledged_by TEXT,                      -- User who acknowledged
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    
    -- Notification tracking
    notified BOOLEAN DEFAULT FALSE,
    notification_sent_at TIMESTAMP,
    itsm_ticket_id TEXT,                      -- External ticket reference
    
    -- Metadata
    raw_data JSONB,                           -- Full alert context
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create hypertable
SELECT create_hypertable('alerts', 'timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes for API queries
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, severity, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_peer ON alerts(peer_addr, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_uuid ON alerts(alert_uuid);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unnotified ON alerts(notified, status) WHERE notified = FALSE;

COMMENT ON TABLE alerts IS 'Final correlated alerts for dashboard and notifications';

-- ============================================================================
-- HELPER TABLES FOR SYSTEM MANAGEMENT
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: system_metrics
-- Purpose: Track system health and performance metrics
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGSERIAL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    metric_name TEXT NOT NULL,                 -- 'ingestion_rate', 'detection_latency', etc.
    metric_value REAL NOT NULL,
    unit TEXT,                                 -- 'records/sec', 'milliseconds', etc.
    component TEXT,                            -- 'ris_collector', 'ml_detector', etc.
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

SELECT create_hypertable('system_metrics', 'timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_system_metrics ON system_metrics(metric_name, component, timestamp DESC);

COMMENT ON TABLE system_metrics IS 'System performance and health metrics';

-- ============================================================================
-- VIEWS FOR EASIER QUERYING
-- ============================================================================

-- ----------------------------------------------------------------------------
-- View: recent_alerts
-- Purpose: Quick access to recent alerts with all relevant information
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW recent_alerts AS
SELECT 
    a.id,
    a.alert_uuid,
    a.timestamp,
    a.alert_type,
    a.severity,
    a.confidence,
    a.peer_addr,
    a.peer_asn,
    a.title,
    a.description,
    a.status,
    a.final_score,
    a.created_at,
    m.ensemble_score as ml_score,
    m.heuristic_reasons,
    COUNT(r.id) as route_monitor_events_count
FROM alerts a
LEFT JOIN ml_results m ON a.ml_result_id = m.id
LEFT JOIN route_monitor_events r ON r.id = ANY(a.route_monitor_event_ids)
GROUP BY a.id, a.alert_uuid, a.timestamp, a.alert_type, a.severity, 
         a.confidence, a.peer_addr, a.peer_asn, a.title, a.description,
         a.status, a.final_score, a.created_at, m.ensemble_score, m.heuristic_reasons
ORDER BY a.timestamp DESC;

COMMENT ON VIEW recent_alerts IS 'Recent alerts with correlated data for dashboard';

-- ----------------------------------------------------------------------------
-- View: alert_summary_hourly
-- Purpose: Hourly statistics for dashboard charts
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW alert_summary_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) as hour,
    severity,
    alert_type,
    COUNT(*) as alert_count,
    AVG(confidence) as avg_confidence,
    AVG(final_score) as avg_score
FROM alerts
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY hour, severity, alert_type
ORDER BY hour DESC;

COMMENT ON VIEW alert_summary_hourly IS 'Hourly alert statistics for visualization';

-- ============================================================================
-- RETENTION POLICIES (TimescaleDB Data Lifecycle Management)
-- ============================================================================

-- Drop old raw data after 30 days (saves storage space)
SELECT add_retention_policy('raw_bgp_data', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep features for 90 days
SELECT add_retention_policy('features', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep ML results for 60 days
SELECT add_retention_policy('ml_results', INTERVAL '60 days', if_not_exists => TRUE);

-- Keep route monitor events for 180 days (longer for compliance)
SELECT add_retention_policy('route_monitor_events', INTERVAL '180 days', if_not_exists => TRUE);

-- Keep alerts for 1 year (important historical data)
SELECT add_retention_policy('alerts', INTERVAL '365 days', if_not_exists => TRUE);

-- Keep system metrics for 30 days
SELECT add_retention_policy('system_metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================================
-- FUNCTIONS FOR DATA PROCESSING
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Function: calculate_ensemble_score
-- Purpose: Helper function to calculate weighted ensemble score
-- Used by: Correlation Engine
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calculate_ensemble_score(
    lstm_score REAL,
    if_score REAL,
    heuristic_score REAL,
    lstm_weight REAL DEFAULT 0.4,
    if_weight REAL DEFAULT 0.3,
    heuristic_weight REAL DEFAULT 0.3
) RETURNS REAL AS $$
BEGIN
    RETURN (
        COALESCE(lstm_score, 0) * lstm_weight +
        COALESCE(if_score, 0) * if_weight +
        COALESCE(heuristic_score, 0) * heuristic_weight
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_ensemble_score IS 'Calculate weighted ensemble score from detector outputs';

-- ----------------------------------------------------------------------------
-- Function: update_alert_status
-- Purpose: Update alert status with timestamp tracking
-- Used by: API when users acknowledge/resolve alerts
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_alert_status(
    p_alert_id BIGINT,
    p_new_status TEXT,
    p_user TEXT DEFAULT NULL,
    p_notes TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE alerts 
    SET 
        status = p_new_status,
        updated_at = NOW(),
        acknowledged_by = CASE WHEN p_new_status = 'acknowledged' THEN p_user ELSE acknowledged_by END,
        acknowledged_at = CASE WHEN p_new_status = 'acknowledged' THEN NOW() ELSE acknowledged_at END,
        resolved_at = CASE WHEN p_new_status = 'resolved' THEN NOW() ELSE resolved_at END,
        resolution_notes = COALESCE(p_notes, resolution_notes)
    WHERE id = p_alert_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_alert_status IS 'Update alert lifecycle status';

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Update alerts.updated_at on any modification
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_alerts_updated_at BEFORE UPDATE ON alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE QUERIES (Documentation)
-- ============================================================================

-- Query 1: Get all critical alerts from last 24 hours
-- SELECT * FROM alerts 
-- WHERE severity = 'critical' 
-- AND timestamp > NOW() - INTERVAL '24 hours'
-- ORDER BY timestamp DESC;

-- Query 2: Get peer anomaly trend
-- SELECT peer_addr, COUNT(*) as alert_count, AVG(final_score) as avg_score
-- FROM alerts
-- WHERE timestamp > NOW() - INTERVAL '7 days'
-- GROUP BY peer_addr
-- ORDER BY alert_count DESC;

-- Query 3: Get detection effectiveness
-- SELECT 
--     alert_type,
--     COUNT(*) as total_alerts,
--     SUM(CASE WHEN status = 'false_positive' THEN 1 ELSE 0 END) as false_positives,
--     AVG(confidence) as avg_confidence
-- FROM alerts
-- GROUP BY alert_type;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'BGP Anomaly Detection Schema Created!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created: 6 main tables + 1 helper table';
    RAISE NOTICE 'Views created: 2 views';
    RAISE NOTICE 'Functions created: 2 functions';
    RAISE NOTICE 'Retention policies: Applied to all time-series tables';
    RAISE NOTICE '========================================';
END $$;
