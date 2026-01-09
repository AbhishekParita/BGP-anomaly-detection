-- Database Triggers and Limits for BGP Monitoring System
-- PostgreSQL triggers to automatically enforce data limits

-- ============================================
-- 1. CREATE TRIGGER FUNCTION FOR ROW LIMIT
-- ============================================

CREATE OR REPLACE FUNCTION enforce_table_limit()
RETURNS TRIGGER AS $$
DECLARE
    row_count INTEGER;
    max_rows INTEGER := 50000;  -- Maximum rows allowed
    delete_count INTEGER;
BEGIN
    -- Count current rows
    EXECUTE format('SELECT COUNT(*) FROM %I.%I', TG_TABLE_SCHEMA, TG_TABLE_NAME) INTO row_count;
    
    -- If we exceed the limit, delete oldest 20% of records
    IF row_count >= max_rows THEN
        delete_count := CAST(max_rows * 0.2 AS INTEGER);
        
        EXECUTE format('
            DELETE FROM %I.%I 
            WHERE id IN (
                SELECT id FROM %I.%I 
                ORDER BY timestamp ASC 
                LIMIT %s
            )',
            TG_TABLE_SCHEMA, TG_TABLE_NAME,
            TG_TABLE_SCHEMA, TG_TABLE_NAME,
            delete_count
        );
        
        RAISE NOTICE 'Table % limit reached. Deleted % oldest records.', TG_TABLE_NAME, delete_count;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 2. APPLY TRIGGER TO bgp_announcements
-- ============================================

DROP TRIGGER IF EXISTS enforce_bgp_announcements_limit ON bgp_announcements;

CREATE TRIGGER enforce_bgp_announcements_limit
    AFTER INSERT ON bgp_announcements
    FOR EACH STATEMENT
    EXECUTE FUNCTION enforce_table_limit();

-- ============================================
-- 3. CREATE AUTOMATIC CLEANUP FUNCTION
-- ============================================

CREATE OR REPLACE FUNCTION cleanup_old_bgp_data(days_to_keep INTEGER DEFAULT 30)
RETURNS TABLE(table_name TEXT, deleted_rows INTEGER) AS $$
BEGIN
    -- Clean bgp_announcements
    DELETE FROM bgp_announcements
    WHERE timestamp < (NOW() - INTERVAL '1 day' * days_to_keep);
    
    GET DIAGNOSTICS deleted_rows = ROW_COUNT;
    table_name := 'bgp_announcements';
    RETURN NEXT;
    
    RAISE NOTICE 'Cleanup complete: Deleted % rows older than % days', deleted_rows, days_to_keep;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 4. CREATE MONITORING VIEW
-- ============================================

CREATE OR REPLACE VIEW bgp_monitoring_stats AS
SELECT
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE status = 'valid') as valid_count,
    COUNT(*) FILTER (WHERE status = 'invalid') as invalid_count,
    COUNT(*) FILTER (WHERE status = 'not-found') as not_found_count,
    MIN(timestamp) as oldest_record,
    MAX(timestamp) as newest_record,
    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as last_hour_count,
    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') as last_day_count,
    pg_size_pretty(pg_total_relation_size('bgp_announcements')) as table_size
FROM bgp_announcements;

-- ============================================
-- 5. CREATE INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX IF NOT EXISTS idx_bgp_timestamp ON bgp_announcements(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bgp_status ON bgp_announcements(status);
CREATE INDEX IF NOT EXISTS idx_bgp_asn ON bgp_announcements(asn);
CREATE INDEX IF NOT EXISTS idx_bgp_prefix ON bgp_announcements(prefix);
CREATE INDEX IF NOT EXISTS idx_bgp_composite ON bgp_announcements(timestamp DESC, status);

-- ============================================
-- 6. CREATE SCHEDULED CLEANUP (PostgreSQL 10+)
-- ============================================

-- Note: Requires pg_cron extension
-- To install: CREATE EXTENSION pg_cron;

-- Schedule daily cleanup at 2 AM
-- SELECT cron.schedule('cleanup-old-bgp-data', '0 2 * * *', 'SELECT cleanup_old_bgp_data(30)');

-- ============================================
-- 7. CREATE FUNCTION TO GET TABLE LIMITS
-- ============================================

CREATE OR REPLACE FUNCTION get_table_info(table_name_param TEXT)
RETURNS TABLE(
    total_rows BIGINT,
    table_size TEXT,
    index_size TEXT,
    total_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*) FROM bgp_announcements) as total_rows,
        pg_size_pretty(pg_table_size(table_name_param::regclass)) as table_size,
        pg_size_pretty(pg_indexes_size(table_name_param::regclass)) as index_size,
        pg_size_pretty(pg_total_relation_size(table_name_param::regclass)) as total_size;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- USAGE EXAMPLES
-- ============================================

-- View current stats:
-- SELECT * FROM bgp_monitoring_stats;

-- Manual cleanup (keep last 30 days):
-- SELECT * FROM cleanup_old_bgp_data(30);

-- Get table information:
-- SELECT * FROM get_table_info('bgp_announcements');

-- View recent records:
-- SELECT * FROM bgp_announcements 
-- WHERE timestamp > NOW() - INTERVAL '1 hour' 
-- ORDER BY timestamp DESC 
-- LIMIT 100;
