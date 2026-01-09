-- ================================================================
-- BGP Anomaly Detection - Database Triggers & Constraints
-- Auto-cleanup triggers to prevent infinite data accumulation
-- ================================================================

-- Configuration: Set maximum records per table
-- Adjust these values based on your storage capacity
DO $$
BEGIN
    -- Create a config table if it doesn't exist
    CREATE TABLE IF NOT EXISTS system_config (
        config_key VARCHAR(50) PRIMARY KEY,
        config_value INTEGER NOT NULL,
        description TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Insert default limits
    INSERT INTO system_config (config_key, config_value, description) 
    VALUES 
        ('max_raw_bgp_records', 100000, 'Maximum records in raw_bgp_data table'),
        ('max_bgp_announcements', 100000, 'Maximum records in bgp_announcements table'),
        ('max_ml_results', 50000, 'Maximum records in ml_results table'),
        ('max_ensemble_results', 50000, 'Maximum records in ensemble_results table'),
        ('retention_days', 30, 'Number of days to retain old records')
    ON CONFLICT (config_key) DO NOTHING;
END $$;

-- ================================================================
-- FUNCTION: Cleanup old records from a table
-- ================================================================
CREATE OR REPLACE FUNCTION cleanup_old_records()
RETURNS TRIGGER AS $$
DECLARE
    retention_days INTEGER;
    delete_count INTEGER;
BEGIN
    -- Get retention period from config
    SELECT config_value INTO retention_days 
    FROM system_config 
    WHERE config_key = 'retention_days';
    
    -- Delete records older than retention period
    EXECUTE format(
        'DELETE FROM %I WHERE timestamp < NOW() - INTERVAL ''%s days''',
        TG_TABLE_NAME, retention_days
    );
    
    GET DIAGNOSTICS delete_count = ROW_COUNT;
    
    IF delete_count > 0 THEN
        RAISE NOTICE 'Cleaned up % old records from %', delete_count, TG_TABLE_NAME;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- FUNCTION: Enforce maximum record limit
-- ================================================================
CREATE OR REPLACE FUNCTION enforce_record_limit()
RETURNS TRIGGER AS $$
DECLARE
    max_records INTEGER;
    current_count INTEGER;
    excess_count INTEGER;
    config_key_name TEXT;
BEGIN
    -- Determine config key based on table name
    config_key_name := 'max_' || TG_TABLE_NAME;
    
    -- Get maximum records limit from config
    SELECT config_value INTO max_records 
    FROM system_config 
    WHERE config_key = config_key_name;
    
    -- If no config found, use default
    IF max_records IS NULL THEN
        max_records := 100000;
    END IF;
    
    -- Check current record count
    EXECUTE format('SELECT COUNT(*) FROM %I', TG_TABLE_NAME) INTO current_count;
    
    -- If over limit, delete oldest records
    IF current_count >= max_records THEN
        excess_count := current_count - max_records + 1000; -- Delete extra 1000 for buffer
        
        EXECUTE format(
            'DELETE FROM %I WHERE timestamp IN (
                SELECT timestamp FROM %I 
                ORDER BY timestamp ASC 
                LIMIT %s
            )',
            TG_TABLE_NAME, TG_TABLE_NAME, excess_count
        );
        
        RAISE NOTICE 'Deleted % oldest records from % (limit: %)', 
            excess_count, TG_TABLE_NAME, max_records;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- TRIGGER: Auto-cleanup for raw_bgp_data
-- ================================================================
DROP TRIGGER IF EXISTS trigger_cleanup_raw_bgp_data ON raw_bgp_data;
CREATE TRIGGER trigger_cleanup_raw_bgp_data
    AFTER INSERT ON raw_bgp_data
    FOR EACH STATEMENT
    EXECUTE FUNCTION cleanup_old_records();

DROP TRIGGER IF EXISTS trigger_limit_raw_bgp_data ON raw_bgp_data;
CREATE TRIGGER trigger_limit_raw_bgp_data
    AFTER INSERT ON raw_bgp_data
    FOR EACH STATEMENT
    EXECUTE FUNCTION enforce_record_limit();

-- ================================================================
-- TRIGGER: Auto-cleanup for bgp_announcements
-- ================================================================
DROP TRIGGER IF EXISTS trigger_cleanup_bgp_announcements ON bgp_announcements;
CREATE TRIGGER trigger_cleanup_bgp_announcements
    AFTER INSERT ON bgp_announcements
    FOR EACH STATEMENT
    EXECUTE FUNCTION cleanup_old_records();

DROP TRIGGER IF EXISTS trigger_limit_bgp_announcements ON bgp_announcements;
CREATE TRIGGER trigger_limit_bgp_announcements
    AFTER INSERT ON bgp_announcements
    FOR EACH STATEMENT
    EXECUTE FUNCTION enforce_record_limit();

-- ================================================================
-- TRIGGER: Auto-cleanup for ml_results (if exists)
-- ================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ml_results') THEN
        DROP TRIGGER IF EXISTS trigger_cleanup_ml_results ON ml_results;
        CREATE TRIGGER trigger_cleanup_ml_results
            AFTER INSERT ON ml_results
            FOR EACH STATEMENT
            EXECUTE FUNCTION cleanup_old_records();
            
        DROP TRIGGER IF EXISTS trigger_limit_ml_results ON ml_results;
        CREATE TRIGGER trigger_limit_ml_results
            AFTER INSERT ON ml_results
            FOR EACH STATEMENT
            EXECUTE FUNCTION enforce_record_limit();
    END IF;
END $$;

-- ================================================================
-- TRIGGER: Auto-cleanup for ensemble_results (if exists)
-- ================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ensemble_results') THEN
        DROP TRIGGER IF EXISTS trigger_cleanup_ensemble_results ON ensemble_results;
        CREATE TRIGGER trigger_cleanup_ensemble_results
            AFTER INSERT ON ensemble_results
            FOR EACH STATEMENT
            EXECUTE FUNCTION cleanup_old_records();
            
        DROP TRIGGER IF EXISTS trigger_limit_ensemble_results ON ensemble_results;
        CREATE TRIGGER trigger_limit_ensemble_results
            AFTER INSERT ON ensemble_results
            FOR EACH STATEMENT
            EXECUTE FUNCTION enforce_record_limit();
    END IF;
END $$;

-- ================================================================
-- INDEXES for better cleanup performance
-- ================================================================
CREATE INDEX IF NOT EXISTS idx_raw_bgp_data_timestamp 
    ON raw_bgp_data(timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_bgp_announcements_timestamp 
    ON bgp_announcements(timestamp DESC);

-- ================================================================
-- VIEW: System statistics
-- ================================================================
CREATE OR REPLACE VIEW system_stats AS
SELECT 
    'raw_bgp_data' as table_name,
    COUNT(*) as record_count,
    MIN(timestamp) as oldest_record,
    MAX(timestamp) as newest_record,
    pg_size_pretty(pg_total_relation_size('raw_bgp_data')) as table_size
FROM raw_bgp_data
UNION ALL
SELECT 
    'bgp_announcements' as table_name,
    COUNT(*) as record_count,
    MIN(timestamp) as oldest_record,
    MAX(timestamp) as newest_record,
    pg_size_pretty(pg_total_relation_size('bgp_announcements')) as table_size
FROM bgp_announcements;

-- ================================================================
-- STORED PROCEDURE: Manual cleanup
-- ================================================================
CREATE OR REPLACE PROCEDURE manual_cleanup()
LANGUAGE plpgsql
AS $$
DECLARE
    retention_days INTEGER;
    deleted_count INTEGER := 0;
BEGIN
    SELECT config_value INTO retention_days 
    FROM system_config 
    WHERE config_key = 'retention_days';
    
    RAISE NOTICE 'Starting manual cleanup (retention: % days)...', retention_days;
    
    -- Cleanup raw_bgp_data
    DELETE FROM raw_bgp_data 
    WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'Deleted % records from raw_bgp_data', deleted_count;
    
    -- Cleanup bgp_announcements
    DELETE FROM bgp_announcements 
    WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'Deleted % records from bgp_announcements', deleted_count;
    
    -- Vacuum tables
    VACUUM ANALYZE raw_bgp_data;
    VACUUM ANALYZE bgp_announcements;
    
    RAISE NOTICE 'Manual cleanup completed!';
END;
$$;

-- ================================================================
-- Grant permissions (adjust as needed)
-- ================================================================
GRANT SELECT ON system_config TO PUBLIC;
GRANT SELECT ON system_stats TO PUBLIC;

-- ================================================================
-- Display configuration
-- ================================================================
SELECT 
    config_key,
    config_value,
    description,
    updated_at
FROM system_config
ORDER BY config_key;

-- Display current statistics
SELECT * FROM system_stats;

-- ================================================================
-- USAGE EXAMPLES:
-- ================================================================
-- 1. View current statistics:
--    SELECT * FROM system_stats;
--
-- 2. Update configuration:
--    UPDATE system_config SET config_value = 50000 
--    WHERE config_key = 'max_raw_bgp_records';
--
-- 3. Manual cleanup:
--    CALL manual_cleanup();
--
-- 4. Check trigger status:
--    SELECT trigger_name, event_manipulation, event_object_table 
--    FROM information_schema.triggers 
--    WHERE trigger_schema = 'public';
-- ================================================================

RAISE NOTICE '✅ Database triggers and constraints installed successfully!';
