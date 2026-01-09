--
-- PostgreSQL database dump
--

-- Dumped from database version 14.5 (Ubuntu 14.5-1.pgdg22.04+1)
-- Dumped by pg_dump version 14.5 (Ubuntu 14.5-1.pgdg22.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: timescaledb; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS timescaledb WITH SCHEMA public;


--
-- Name: EXTENSION timescaledb; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION timescaledb IS 'Enables scalable inserts and complex queries for time-series data';


--
-- Name: pg_cron; Type: EXTENSION; Schema: -; Owner: -
--

--CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA public;


--
-- Name: EXTENSION pg_cron; Type: COMMENT; Schema: -; Owner: -
--

--COMMENT ON EXTENSION pg_cron IS 'Job scheduler for PostgreSQL';


--
-- Name: timescaledb_toolkit; Type: EXTENSION; Schema: -; Owner: -
--

--CREATE EXTENSION IF NOT EXISTS timescaledb_toolkit WITH SCHEMA public;


--
-- Name: EXTENSION timescaledb_toolkit; Type: COMMENT; Schema: -; Owner: -
--

--COMMENT ON EXTENSION timescaledb_toolkit IS 'Library of analytical hyperfunctions, time-series pipelining, and other SQL utilities';


--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';


--
-- Name: pgrouting; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgrouting WITH SCHEMA public;


--
-- Name: EXTENSION pgrouting; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pgrouting IS 'pgRouting Extension';


--
-- Name: ls_mpls_proto_mask; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.ls_mpls_proto_mask AS ENUM (
    'LDP',
    'RSVP-TE',
    ''
);


--
-- Name: ls_proto; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.ls_proto AS ENUM (
    'IS-IS_L1',
    'IS-IS_L2',
    'OSPFv2',
    'Direct',
    'Static',
    'OSPFv3',
    ''
);


--
-- Name: opstate; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.opstate AS ENUM (
    'up',
    'down',
    ''
);


--
-- Name: ospf_route_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.ospf_route_type AS ENUM (
    'Intra',
    'Inter',
    'Ext-1',
    'Ext-2',
    'NSSA-1',
    'NSSA-2',
    ''
);


--
-- Name: user_role; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.user_role AS ENUM (
    'admin',
    'oper',
    ''
);


--
-- Name: acknowledge_anomaly(integer, character varying); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.acknowledge_anomaly(anomaly_id integer, acknowledged_by_user character varying) RETURNS boolean
    LANGUAGE plpgsql
    AS $$
BEGIN
    UPDATE anomaly_events
    SET 
        acknowledged = TRUE,
        acknowledged_at = NOW(),
        acknowledged_by = acknowledged_by_user,
        updated_at = NOW()
    WHERE id = anomaly_id;
    
    RETURN FOUND;
END;
$$;


--
-- Name: find_geo_ip(inet); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.find_geo_ip(find_ip inet) RETURNS inet
    LANGUAGE plpgsql
    AS $_$
	DECLARE
	        geo_ip_prefix inet := NULL;
	BEGIN

	    -- Use execute for better performance - http://blog.endpoint.com/2008/12/why-is-my-function-slow.html
	    EXECUTE 'SELECT ip
		    FROM geo_ip
	        WHERE ip && $1
	        ORDER BY ip desc
	        LIMIT 1' INTO geo_ip_prefix USING find_ip;

		RETURN geo_ip_prefix;
	END
$_$;


--
-- Name: get_next_router_index(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_next_router_index() RETURNS smallint
    LANGUAGE plpgsql
    AS $$
DECLARE
	_idx smallint := 0;
	_prev_idx smallint := 0;
BEGIN

	FOR _idx IN SELECT index FROM routers ORDER BY index LOOP
		IF (_prev_idx = 0) THEN
			_prev_idx := _idx;
			CONTINUE;

		ELSIF ( (_prev_idx + 1) != _idx) THEN
			-- Found available index
			RETURN _prev_idx + 1;
		END IF;

		_prev_idx := _idx;
	END LOOP;

	RETURN _prev_idx + 1;
END;
$$;


--
-- Name: get_peer_baseline(inet, bigint, character varying, character varying); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_peer_baseline(p_peer_addr inet, p_peer_as bigint, p_region character varying DEFAULT NULL::character varying, p_tenant_id character varying DEFAULT NULL::character varying) RETURNS integer
    LANGUAGE plpgsql
    AS $$
DECLARE
    v_baseline INT;
BEGIN
    -- Priority order: peer > region > tenant > global
    
    -- 1. Check peer-specific config
    SELECT baseline_threshold INTO v_baseline
    FROM bmp_baseline_config
    WHERE config_type = 'peer'
      AND peer_addr = p_peer_addr
      AND (peer_as = p_peer_as OR peer_as IS NULL)
      AND enabled = true
    LIMIT 1;
    
    IF v_baseline IS NOT NULL THEN
        RETURN v_baseline;
    END IF;
    
    -- 2. Check region-specific config
    IF p_region IS NOT NULL THEN
        SELECT baseline_threshold INTO v_baseline
        FROM bmp_baseline_config
        WHERE config_type = 'region'
          AND region = p_region
          AND enabled = true
        LIMIT 1;
        
        IF v_baseline IS NOT NULL THEN
            RETURN v_baseline;
        END IF;
    END IF;
    
    -- 3. Check tenant-specific config
    IF p_tenant_id IS NOT NULL THEN
        SELECT baseline_threshold INTO v_baseline
        FROM bmp_baseline_config
        WHERE config_type = 'tenant'
          AND tenant_id = p_tenant_id
          AND enabled = true
        LIMIT 1;
        
        IF v_baseline IS NOT NULL THEN
            RETURN v_baseline;
        END IF;
    END IF;
    
    -- 4. Fall back to global config
    SELECT baseline_threshold INTO v_baseline
    FROM bmp_baseline_config
    WHERE config_type = 'global'
      AND enabled = true
    LIMIT 1;
    
    RETURN COALESCE(v_baseline, 100); -- Default to 100 if no config found
END;
$$;


--
-- Name: purge_global_ip_rib(interval); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.purge_global_ip_rib(older_than_time interval DEFAULT '04:00:00'::interval) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
	-- delete old withdrawn prefixes that we don't want to track anymore
	DELETE FROM global_ip_rib where iswithdrawn = true and timestamp < now () - older_than_time;

END;
$$;


--
-- Name: refresh_anomaly_views(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.refresh_anomaly_views() RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY v_severity_heatmap;
    REFRESH MATERIALIZED VIEW CONCURRENTLY v_anomaly_type_summary;
END;
$$;


--
-- Name: show_table_info(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.show_table_info() RETURNS TABLE(oid oid, table_schema name, table_name name, row_estimate real, total_bytes bigint, index_bytes bigint, toast_bytes bigint, table_bytes bigint, total character varying, index character varying, toast character varying, table_value character varying)
    LANGUAGE sql
    AS $$
	    SELECT *, pg_size_pretty(total_bytes) AS total,
                pg_size_pretty(index_bytes) AS INDEX,
                pg_size_pretty(toast_bytes) AS toast,
                pg_size_pretty(table_bytes) AS table_value
		  FROM (
			  SELECT *, total_bytes-index_bytes-COALESCE(toast_bytes,0) AS table_bytes FROM (
			      SELECT c.oid,nspname AS table_schema, relname AS TABLE_NAME,
			              c.reltuples AS row_estimate,
			              pg_total_relation_size(c.oid) AS total_bytes,
			              pg_indexes_size(c.oid) AS index_bytes,
			              pg_total_relation_size(reltoastrelid) AS toast_bytes
			          FROM pg_class c
			          LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
			          WHERE relkind = 'r'
			  ) a
		) a;
	$$;


--
-- Name: sync_global_ip_rib(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.sync_global_ip_rib() RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
	execution_start timestamptz  := clock_timestamp();
	insert_count    int;
	start_time timestamptz := now();
BEGIN

	raise INFO 'Start time       : %', execution_start;

	INSERT INTO global_ip_rib (prefix,prefix_len,recv_origin_as,
	                           iswithdrawn,timestamp,first_added_timestamp,num_peers,advertising_peers,withdrawn_peers)

	SELECT r.prefix,
	       max(r.prefix_len),
	       r.origin_as,
	       bool_and(r.iswithdrawn)                                             as isWithdrawn,
	       max(r.timestamp),
	       min(r.first_added_timestamp),
	       count(distinct r.peer_hash_id)                                      as total_peers,
	       count(distinct r.peer_hash_id) FILTER (WHERE r.iswithdrawn = False) as advertising_peers,
	       count(distinct r.peer_hash_id) FILTER (WHERE r.iswithdrawn = True)  as withdrawn_peers
	FROM ip_rib r
	WHERE origin_as != 23456
	GROUP BY r.prefix, r.origin_as
	ON CONFLICT (prefix,recv_origin_as)
		DO UPDATE SET timestamp=excluded.timestamp,
		              first_added_timestamp=excluded.first_added_timestamp,
		              iswithdrawn=excluded.iswithdrawn,
		              num_peers=excluded.num_peers,
		              advertising_peers=excluded.advertising_peers,
		              withdrawn_peers=excluded.withdrawn_peers;

	GET DIAGNOSTICS insert_count = row_count;
	raise INFO 'Rows updated   : %', insert_count;
	raise INFO 'Duration       : %', clock_timestamp() - execution_start;
	raise INFO 'Completion time: %', clock_timestamp();

	-- Update IRR
	raise INFO '-> Updating IRR info';
	UPDATE global_ip_rib r SET
		                       irr_origin_as=i.origin_as,
		                       irr_source=i.source,
		                       irr_descr=i.descr
	FROM info_route i
	WHERE  i.prefix = r.prefix;

	GET DIAGNOSTICS insert_count = row_count;
	raise INFO 'Rows updated   : %', insert_count;
	raise INFO 'Duration       : %', clock_timestamp() - execution_start;
	raise INFO 'Completion time: %', clock_timestamp();


	-- Update RPKI entries - Limit query to only update what has changed in interval time
	--    NOTE: The global_ip_rib table should have current times when first run (new table).
	--          This will result in this query taking a while. After first run, it shouldn't take
	--          as long.
	raise INFO '-> Updating RPKI info';
	UPDATE global_ip_rib r SET rpki_origin_as=p.origin_as
	FROM rpki_validator p
	WHERE
			p.prefix >>= r.prefix
	  AND r.prefix_len >= p.prefix_len
	  AND r.prefix_len <= p.prefix_len_max;

	GET DIAGNOSTICS insert_count = row_count;
	raise INFO 'Rows updated   : %', insert_count;
	raise INFO 'Duration       : %', clock_timestamp() - execution_start;


	raise INFO 'Completion time: %', clock_timestamp();

END;
$$;


--
-- Name: t_bgp_peers(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_bgp_peers() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	IF (new.peer_addr = '0.0.0.0' AND new.peer_bgp_id = '0.0.0.0') THEN
		SELECT r.name,r.ip_address INTO new.name,new.peer_bgp_id
			FROM routers r WHERE r.hash_id = new.router_hash_id;
	END IF;

	SELECT find_geo_ip(new.peer_addr) INTO new.geo_ip_start;

	IF (new.state = 'up') THEN
		INSERT INTO peer_event_log (state,peer_hash_id,local_ip,local_bgp_id,local_port,local_hold_time,
                                    local_asn,remote_port,remote_hold_time,
                                    sent_capabilities,recv_capabilities,geo_ip_start,timestamp)
                VALUES (new.state,new.hash_id,new.local_ip,new.local_bgp_id,new.local_port,new.local_hold_time,
                        new.local_asn,new.remote_port,new.remote_hold_time,
                        new.sent_capabilities,new.recv_capabilities,new.geo_ip_start, new.timestamp);
	ELSE
		-- Updated using old values since those are not in the down state
		INSERT INTO peer_event_log (state,peer_hash_id,local_ip,local_bgp_id,local_port,local_hold_time,
                                    local_asn,remote_port,remote_hold_time,
                                    sent_capabilities,recv_capabilities,bmp_reason,bgp_err_code,
                                    bgp_err_subcode,error_text,timestamp)
                VALUES (new.state,new.hash_id,new.local_ip,new.local_bgp_id,new.local_port,new.local_hold_time,
                        new.local_asn,new.remote_port,new.remote_hold_time,
                        new.sent_capabilities,new.recv_capabilities,new.bmp_reason,new.bgp_err_code,
                        new.bgp_err_subcode,new.error_text,new.timestamp);

	END IF;

	RETURN NEW;
END;
$$;


--
-- Name: t_ip_rib_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_ip_rib_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	IF (new.isWithdrawn) THEN
		INSERT INTO ip_rib_log (isWithdrawn,prefix,prefix_len,base_attr_hash_id,peer_hash_id,origin_as,timestamp)
		VALUES (true,new.prefix,new.prefix_len,old.base_attr_hash_id,new.peer_hash_id,
		        old.origin_as,new.timestamp);
	ELSE
		INSERT INTO ip_rib_log (isWithdrawn,prefix,prefix_len,base_attr_hash_id,peer_hash_id,origin_as,timestamp)
		VALUES (false,new.prefix,new.prefix_len,new.base_attr_hash_id,new.peer_hash_id,
		        new.origin_as,new.timestamp);
	END IF;

	RETURN NEW;
END;
$$;


--
-- Name: t_l3vpn_rib_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_l3vpn_rib_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	IF (new.isWithdrawn) THEN
		INSERT INTO l3vpn_rib_log (isWithdrawn,prefix,prefix_len,base_attr_hash_id,peer_hash_id,origin_as,timestamp,
		                           rd,ext_community_list)
		VALUES (true,new.prefix,new.prefix_len,old.base_attr_hash_id,new.peer_hash_id,
		        old.origin_as,new.timestamp,old.rd,old.ext_community_list);
	ELSE
		INSERT INTO l3vpn_rib_log (isWithdrawn,prefix,prefix_len,base_attr_hash_id,peer_hash_id,origin_as,timestamp,
		                           rd,ext_community_list)
		VALUES (false,new.prefix,new.prefix_len,new.base_attr_hash_id,new.peer_hash_id,
		        new.origin_as,new.timestamp,new.rd,new.ext_community_list);
	END IF;

	RETURN NEW;
END;
$$;


--
-- Name: t_ls_links_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_ls_links_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	-- Only update
	-- Add record to log table if there is a change
	IF ((new.isWithdrawn <> old.isWithdrawn) OR (not new.isWithdrawn AND new.base_attr_hash_id <> old.base_attr_hash_id)) THEN
		IF (new.isWithdrawn) THEN
			INSERT INTO ls_links_log (hash_id, peer_hash_id, base_attr_hash_id, seq, mt_id, interface_addr, neighbor_addr,
					isipv4, protocol, local_link_id, remote_link_id, local_node_hash_id, remote_node_hash_id, admin_group,
					max_link_bw, max_resv_bw, unreserved_bw, te_def_metric, protection_type,
					mpls_proto_mask, igp_metric, srlg, name, local_igp_router_id, local_router_id,
					remote_igp_router_id, remote_router_id, local_asn, remote_asn,
					peer_node_sid, sr_adjacency_sids,
					iswithdrawn)
				VALUES (new.hash_id, new.peer_hash_id, old.base_attr_hash_id, new.seq, old.mt_id, old.interface_addr, old.neighbor_addr,
					old.isipv4, old.protocol, old.local_link_id, old.remote_link_id, old.local_node_hash_id, old.remote_node_hash_id, old.admin_group,
					old.max_link_bw, old.max_resv_bw, old.unreserved_bw, old.te_def_metric, old.protection_type,
					old.mpls_proto_mask, old.igp_metric, old.srlg, old.name, old.local_igp_router_id, old.local_router_id,
					old.remote_igp_router_id, old.remote_router_id, old.local_asn, old.remote_asn,
					old.peer_node_sid, old.sr_adjacency_sids,
					true);
		ELSE
				INSERT INTO ls_links_log (hash_id, peer_hash_id, base_attr_hash_id, seq, mt_id, interface_addr, neighbor_addr,
					isipv4, protocol, local_link_id, remote_link_id, local_node_hash_id, remote_node_hash_id, admin_group,
					max_link_bw, max_resv_bw, unreserved_bw, te_def_metric, protection_type,
					mpls_proto_mask, igp_metric, srlg, name, local_igp_router_id, local_router_id,
					remote_igp_router_id, remote_router_id, local_asn, remote_asn,
					peer_node_sid, sr_adjacency_sids,
					iswithdrawn)
				VALUES (new.hash_id, new.peer_hash_id, new.base_attr_hash_id, new.seq, new.mt_id, new.interface_addr, new.neighbor_addr,
					new.isipv4, new.protocol, new.local_link_id, new.remote_link_id, new.local_node_hash_id, new.remote_node_hash_id, new.admin_group,
					new.max_link_bw, new.max_resv_bw, new.unreserved_bw, new.te_def_metric, new.protection_type,
					new.mpls_proto_mask, new.igp_metric, new.srlg, new.name, new.local_igp_router_id, new.local_router_id,
					new.remote_igp_router_id, new.remote_router_id, new.local_asn, new.remote_asn,
					new.peer_node_sid, new.sr_adjacency_sids,
					false);
		END IF;
	END IF;

	RETURN NEW;
END;
$$;


--
-- Name: t_ls_nodes_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_ls_nodes_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	-- Only update
	-- Add record to log table if there is a change
	IF ((new.isWithdrawn <> old.isWithdrawn) OR (not new.isWithdrawn AND new.base_attr_hash_id <> old.base_attr_hash_id)) THEN
		IF (new.isWithdrawn) THEN
			INSERT INTO ls_nodes_log (hash_id, peer_hash_id, base_attr_hash_id, seq, asn, bgp_ls_id, igp_router_id,
			        ospf_area_id, protocol, router_id, isis_area_id, flags, name, mt_ids, sr_capabilities,
			        iswithdrawn)
				VALUES (new.hash_id, new.peer_hash_id, old.base_attr_hash_id, new.seq, old.asn, old.bgp_ls_id, old.igp_router_id,
					old.ospf_area_id, old.protocol, old.router_id, old.isis_area_id, old.flags, old.name, old.mt_ids, old.sr_capabilities,
					true);
		ELSE
			INSERT INTO ls_nodes_log (hash_id, peer_hash_id, base_attr_hash_id, seq, asn, bgp_ls_id, igp_router_id,
			        ospf_area_id, protocol, router_id, isis_area_id, flags, name, mt_ids, sr_capabilities,
			        iswithdrawn)
				VALUES (new.hash_id, new.peer_hash_id, new.base_attr_hash_id, new.seq, old.asn, new.bgp_ls_id, new.igp_router_id,
					new.ospf_area_id, new.protocol, new.router_id, new.isis_area_id, new.flags, new.name, new.mt_ids, new.sr_capabilities,
					false);
		END IF;
	END IF;

	RETURN NEW;
END;
$$;


--
-- Name: t_ls_prefixes_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_ls_prefixes_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	-- Only update
	-- Add record to log table if there is a change
	IF ((new.isWithdrawn <> old.isWithdrawn) OR (not new.isWithdrawn AND new.base_attr_hash_id <> old.base_attr_hash_id)) THEN
		IF (new.isWithdrawn) THEN
			INSERT INTO ls_prefixes_log (hash_id, peer_hash_id, base_attr_hash_id, seq,
					local_node_hash_id, mt_id, protocol, prefix, prefix_len, ospf_route_type,
					igp_flags, isipv4, route_tag, ext_route_tag, metric, ospf_fwd_addr,
					sr_prefix_sids, iswithdrawn)
				VALUES (new.hash_id, new.peer_hash_id, old.base_attr_hash_id, new.seq,
					old.local_node_hash_id, old.mt_id, old.protocol, old.prefix, old.prefix_len, old.ospf_route_type,
					old.igp_flags, old.isipv4, old.route_tag, old.ext_route_tag, old.metric, old.ospf_fwd_addr,
					old.sr_prefix_sids, true);
		ELSE
			INSERT INTO ls_prefixes_log (hash_id, peer_hash_id, base_attr_hash_id, seq,
					local_node_hash_id, mt_id, protocol, prefix, prefix_len, ospf_route_type,
					igp_flags, isipv4, route_tag, ext_route_tag, metric, ospf_fwd_addr,
					sr_prefix_sids, iswithdrawn)
				VALUES (new.hash_id, new.peer_hash_id, new.base_attr_hash_id, new.seq,
					new.local_node_hash_id, new.mt_id, new.protocol, new.prefix, new.prefix_len, new.ospf_route_type,
					new.igp_flags, new.isipv4, new.route_tag, new.ext_route_tag, new.metric, new.ospf_fwd_addr,
					new.sr_prefix_sids, false);

		END IF;
	END IF;

	RETURN NEW;
END;
$$;


--
-- Name: t_routers_insert(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_routers_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	SELECT find_geo_ip(new.ip_address) INTO new.geo_ip_start;

	RETURN NEW;
END;
$$;


--
-- Name: t_routers_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.t_routers_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
	SELECT find_geo_ip(new.ip_address) INTO new.geo_ip_start;

	RETURN NEW;
END;
$$;


--
-- Name: update_chg_stats(interval); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_chg_stats(int_window interval) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
  -- bypeer updates
  INSERT INTO stats_chg_bypeer (interval_time, peer_hash_id, withdraws,updates)
	SELECT
	       to_timestamp((extract(epoch from timestamp)::bigint / 60)::bigint * 60) at time zone 'utc' as IntervalTime,
	       peer_hash_id,
	       count(case WHEN ip_rib_log.iswithdrawn = true THEN 1 ELSE null END) as withdraws,
	       count(case WHEN ip_rib_log.iswithdrawn = false THEN 1 ELSE null END) as updates
	     FROM ip_rib_log
	     WHERE timestamp >= to_timestamp((extract(epoch from now())::bigint / 60)::bigint * 60) at time zone 'utc' - int_window
	           AND timestamp < to_timestamp((extract(epoch from now())::bigint / 60)::bigint * 60) at time zone 'utc'    -- current minute
	     GROUP BY IntervalTime,peer_hash_id
	ON CONFLICT (interval_time,peer_hash_id) DO UPDATE
		SET updates=excluded.updates, withdraws=excluded.withdraws;

  -- byasn updates
  INSERT INTO stats_chg_byasn (interval_time, peer_hash_id, origin_as,withdraws,updates)
	SELECT
	       to_timestamp((extract(epoch from timestamp)::bigint / 60)::bigint * 60) at time zone 'utc' as IntervalTime,
	       peer_hash_id,origin_as,
	       count(case WHEN ip_rib_log.iswithdrawn = true THEN 1 ELSE null END) as withdraws,
	       count(case WHEN ip_rib_log.iswithdrawn = false THEN 1 ELSE null END) as updates
	     FROM ip_rib_log
	     WHERE timestamp >= to_timestamp((extract(epoch from now())::bigint / 60)::bigint * 60) at time zone 'utc' - int_window
	           AND timestamp < to_timestamp((extract(epoch from now())::bigint / 60)::bigint * 60) at time zone 'utc'   -- current minute
	     GROUP BY IntervalTime,peer_hash_id,origin_as
	ON CONFLICT (interval_time,peer_hash_id,origin_as) DO UPDATE
		SET updates=excluded.updates, withdraws=excluded.withdraws;

  -- byprefix updates
  INSERT INTO stats_chg_byprefix (interval_time, peer_hash_id, prefix, prefix_len, withdraws,updates)
	SELECT
	       to_timestamp((extract(epoch from timestamp)::bigint / 120)::bigint * 120) at time zone 'utc' as IntervalTime,
	       peer_hash_id,prefix,prefix_len,
	       count(case WHEN ip_rib_log.iswithdrawn = true THEN 1 ELSE null END) as withdraws,
	       count(case WHEN ip_rib_log.iswithdrawn = false THEN 1 ELSE null END) as updates
	     FROM ip_rib_log
	     WHERE timestamp >= to_timestamp((extract(epoch from now())::bigint / 120)::bigint * 120) at time zone 'utc' - int_window
	           AND timestamp < to_timestamp((extract(epoch from now())::bigint / 120)::bigint * 120) at time zone 'utc'   -- current minute
	     GROUP BY IntervalTime,peer_hash_id,prefix,prefix_len
	ON CONFLICT (interval_time,peer_hash_id,prefix) DO UPDATE
		SET updates=excluded.updates, withdraws=excluded.withdraws;

END;
$$;


--
-- Name: update_global_ip_rib(interval); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_global_ip_rib(max_interval interval DEFAULT '02:00:00'::interval) RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
	execution_start timestamptz  := clock_timestamp();
	insert_count    int;
	start_time timestamptz := now();
BEGIN

	select time_bucket('5 minutes', timestamp - interval '5 minute') INTO start_time
	FROM global_ip_rib order by timestamp desc limit 1;

	IF start_time is null THEN
		start_time = time_bucket('5 minutes', now() - max_interval);
		raise INFO '-> Last query time is null, setting last query time within %', max_interval;
	ELSIF start_time < now() - max_interval THEN
		start_time = time_bucket('5 minutes', now() - max_interval);
		raise INFO '-> Last query time is greater than max % time, setting last query time', max_interval;
	ELSIF start_time > now() THEN
		start_time = time_bucket('5 minutes', now() - interval '15 minutes');
		raise INFO '-> Last query time is greater than current time, setting last query time to past 15 minutes';
	END IF;

	raise INFO 'Start time       : %', execution_start;
	raise INFO 'Last Query Time  : %', start_time;

	raise INFO '-> Updating changed prefixes ...';

	insert_count = 0;

	INSERT INTO global_ip_rib (prefix,prefix_len,recv_origin_as,
	                           iswithdrawn,timestamp,first_added_timestamp,num_peers,advertising_peers,withdrawn_peers)

	SELECT r.prefix,
	       max(r.prefix_len),
	       r.origin_as,
	       bool_and(r.iswithdrawn)                                             as isWithdrawn,
	       max(r.timestamp),
	       min(r.first_added_timestamp),
	       count(distinct r.peer_hash_id)                                      as total_peers,
	       count(distinct r.peer_hash_id) FILTER (WHERE r.iswithdrawn = False) as advertising_peers,
	       count(distinct r.peer_hash_id) FILTER (WHERE r.iswithdrawn = True)  as withdrawn_peers
	FROM ip_rib r
	WHERE
		(timestamp >= start_time OR first_added_timestamp >= start_time)
	  AND origin_as != 23456
	GROUP BY r.prefix, r.origin_as
	ON CONFLICT (prefix,recv_origin_as)
		DO UPDATE SET timestamp=excluded.timestamp,
		              first_added_timestamp=excluded.first_added_timestamp,
		              iswithdrawn=excluded.iswithdrawn,
		              num_peers=excluded.num_peers,
		              advertising_peers=excluded.advertising_peers,
		              withdrawn_peers=excluded.withdrawn_peers;

	GET DIAGNOSTICS insert_count = row_count;
	raise INFO 'Rows updated   : %', insert_count;
	raise INFO 'Duration       : %', clock_timestamp() - execution_start;
	raise INFO 'Completion time: %', clock_timestamp();

	-- Update IRR
	raise INFO '-> Updating IRR info';
	UPDATE global_ip_rib r SET
		                       irr_origin_as=i.origin_as,
		                       irr_source=i.source,
		                       irr_descr=i.descr
	FROM info_route i
	WHERE  r.timestamp >= start_time and i.prefix = r.prefix;

	GET DIAGNOSTICS insert_count = row_count;
	raise INFO 'Rows updated   : %', insert_count;
	raise INFO 'Duration       : %', clock_timestamp() - execution_start;
	raise INFO 'Completion time: %', clock_timestamp();


	-- Update RPKI entries - Limit query to only update what has changed in interval time
	--    NOTE: The global_ip_rib table should have current times when first run (new table).
	--          This will result in this query taking a while. After first run, it shouldn't take
	--          as long.
	raise INFO '-> Updating RPKI info';
	UPDATE global_ip_rib r SET rpki_origin_as=p.origin_as
	FROM rpki_validator p
	WHERE r.timestamp >= start_time
	  AND p.prefix >>= r.prefix
	  AND r.prefix_len >= p.prefix_len
	  AND r.prefix_len <= p.prefix_len_max;

	GET DIAGNOSTICS insert_count = row_count;
	raise INFO 'Rows updated   : %', insert_count;
	raise INFO 'Duration       : %', clock_timestamp() - execution_start;


	raise INFO 'Completion time: %', clock_timestamp();

END;
$$;


--
-- Name: update_l3vpn_chg_stats(interval); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_l3vpn_chg_stats(int_window interval) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
	-- bypeer updates
	INSERT INTO stats_l3vpn_chg_bypeer (interval_time, peer_hash_id, withdraws,updates)
	SELECT
		time_bucket(int_window, now() - int_window) as IntervalTime,
		peer_hash_id,
		count(case WHEN l3vpn_rib_log.iswithdrawn = true THEN 1 ELSE null END) as withdraws,
		count(case WHEN l3vpn_rib_log.iswithdrawn = false THEN 1 ELSE null END) as updates
	FROM l3vpn_rib_log
	WHERE timestamp >= time_bucket(int_window, now() - int_window)
	  AND timestamp < time_bucket(int_window, now())
	GROUP BY IntervalTime,peer_hash_id
	ON CONFLICT (interval_time,peer_hash_id) DO UPDATE
		SET updates=excluded.updates, withdraws=excluded.withdraws;

	-- byrd updates
	INSERT INTO stats_l3vpn_chg_byrd (interval_time, peer_hash_id, rd,withdraws,updates)
	SELECT
		time_bucket(int_window, now() - int_window) as IntervalTime,
		peer_hash_id,rd,
		count(case WHEN l3vpn_rib_log.iswithdrawn = true THEN 1 ELSE null END) as withdraws,
		count(case WHEN l3vpn_rib_log.iswithdrawn = false THEN 1 ELSE null END) as updates
	FROM l3vpn_rib_log
	WHERE timestamp >= time_bucket(int_window, now() - int_window)
	  AND timestamp < time_bucket(int_window, now())
	GROUP BY IntervalTime,peer_hash_id,rd
	ON CONFLICT (interval_time,peer_hash_id,rd) DO UPDATE
		SET updates=excluded.updates, withdraws=excluded.withdraws;

	-- byprefix updates
	INSERT INTO stats_l3vpn_chg_byprefix (interval_time, peer_hash_id, prefix, prefix_len, rd, withdraws,updates)
	SELECT
		time_bucket(int_window, now() - int_window) as IntervalTime,
		peer_hash_id,prefix,prefix_len,rd,
		count(case WHEN l3vpn_rib_log.iswithdrawn = true THEN 1 ELSE null END) as withdraws,
		count(case WHEN l3vpn_rib_log.iswithdrawn = false THEN 1 ELSE null END) as updates
	FROM l3vpn_rib_log
	WHERE timestamp >= time_bucket(int_window, now() - int_window)
	  AND timestamp < time_bucket(int_window, now())
	GROUP BY IntervalTime,peer_hash_id,prefix,prefix_len,rd
	ON CONFLICT (interval_time,peer_hash_id,prefix,rd) DO UPDATE
		SET updates=excluded.updates, withdraws=excluded.withdraws;

END;
$$;


--
-- Name: update_origin_stats(interval); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_origin_stats(int_time interval DEFAULT '00:30:00'::interval) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN

    -- Origin stats (originated v4/v6 with IRR and RPKI counts)
	INSERT INTO stats_ip_origins (interval_time,asn,v4_prefixes,v6_prefixes,
	                              v4_with_rpki,v6_with_rpki,v4_with_irr,v6_with_irr)
	SELECT to_timestamp((extract(epoch from now())::bigint / 3600)::bigint * 3600),
	       recv_origin_as,
	       sum(case when family(prefix) = 4 THEN 1 ELSE 0 END) as v4_prefixes,
	       sum(case when family(prefix) = 6 THEN 1 ELSE 0 END) as v6_prefixes,
	       sum(case when rpki_origin_as > 0 and family(prefix) = 4 THEN 1 ELSE 0 END) as v4_with_rpki,
	       sum(case when rpki_origin_as > 0 and family(prefix) = 6 THEN 1 ELSE 0 END) as v6_with_rpki,
	       sum(case when irr_origin_as > 0 and family(prefix) = 4 THEN 1 ELSE 0 END) as v4_with_irr,
	       sum(case when irr_origin_as > 0 and family(prefix) = 6 THEN 1 ELSE 0 END) as v6_with_irr
	FROM global_ip_rib
	GROUP BY recv_origin_as
	ON CONFLICT (interval_time,asn) DO UPDATE SET v4_prefixes=excluded.v4_prefixes,
	                                              v6_prefixes=excluded.v6_prefixes,
	                                              v4_with_rpki=excluded.v4_with_rpki,
	                                              v6_with_rpki=excluded.v6_with_rpki,
	                                              v4_with_irr=excluded.v4_with_irr,
	                                              v6_with_irr=excluded.v6_with_irr;


END;
$$;


--
-- Name: update_peer_rib_counts(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_peer_rib_counts() RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
     -- Per peer rib counts - every 15 minutes
     INSERT INTO stats_peer_rib (interval_time,peer_hash_id,v4_prefixes,v6_prefixes)
       SELECT  time_bucket('15 minutes', now()),
             peer_hash_id,
             sum(CASE WHEN isIPv4 = true THEN 1 ELSE 0 END) AS v4_prefixes,
             sum(CASE WHEN isIPv4 = false THEN 1 ELSE 0 END) as v6_prefixes
         FROM ip_rib
         WHERE isWithdrawn = false
         GROUP BY peer_hash_id
       ON CONFLICT (interval_time,peer_hash_id) DO UPDATE SET v4_prefixes=excluded.v4_prefixes,
             v6_prefixes=excluded.v6_prefixes;
END;
$$;


--
-- Name: update_peer_update_counts(integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_peer_update_counts(interval_secs integer) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
     -- Per peer update counts for interval
     INSERT INTO stats_peer_update_counts (interval_time,peer_hash_id,
                        advertise_avg,advertise_min,advertise_max,
                        withdraw_avg,withdraw_min,withdraw_max)
       SELECT to_timestamp((extract(epoch from now())::bigint / interval_secs)::bigint * interval_secs),
             peer_hash_id,
             avg(updates), min(updates), max(updates),
             avg(withdraws), min(withdraws), max(withdraws)
         FROM stats_chg_bypeer
         WHERE interval_time >= now() - (interval_secs::text || ' seconds')::interval
         GROUP BY peer_hash_id
       ON CONFLICT (interval_time,peer_hash_id) DO UPDATE SET advertise_avg=excluded.advertise_avg,
             advertise_min=excluded.advertise_min,
             advertise_max=excluded.advertise_max,
             withdraw_avg=excluded.withdraw_avg,
             withdraw_min=excluded.withdraw_min,
             withdraw_max=excluded.withdraw_max;
END;
$$;


--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


SET default_table_access_method = heap;

--
-- Name: _compressed_hypertable_863; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_863 (
    id _timescaledb_internal.compressed_data,
    base_attr_hash_id _timescaledb_internal.compressed_data,
    "timestamp" _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    prefix inet,
    prefix_len _timescaledb_internal.compressed_data,
    origin_as bigint,
    iswithdrawn _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN id SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN base_attr_hash_id SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN base_attr_hash_id SET STORAGE EXTENDED;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN "timestamp" SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN prefix SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN prefix_len SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN origin_as SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN iswithdrawn SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN iswithdrawn SET STORAGE EXTENDED;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_863 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_868; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_868 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    updates _timescaledb_internal.compressed_data,
    withdraws _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN updates SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN withdraws SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_868 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_870; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_870 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    origin_as bigint,
    updates _timescaledb_internal.compressed_data,
    withdraws _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN origin_as SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN updates SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN withdraws SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_870 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_872; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_872 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    prefix inet,
    prefix_len _timescaledb_internal.compressed_data,
    updates _timescaledb_internal.compressed_data,
    withdraws _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN prefix SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN prefix_len SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN updates SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN withdraws SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_872 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_874; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_874 (
    id _timescaledb_internal.compressed_data,
    interval_time _timescaledb_internal.compressed_data,
    asn bigint,
    v4_prefixes _timescaledb_internal.compressed_data,
    v6_prefixes _timescaledb_internal.compressed_data,
    v4_with_rpki _timescaledb_internal.compressed_data,
    v6_with_rpki _timescaledb_internal.compressed_data,
    v4_with_irr _timescaledb_internal.compressed_data,
    v6_with_irr _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN id SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN asn SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN v4_prefixes SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN v6_prefixes SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN v4_with_rpki SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN v6_with_rpki SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN v4_with_irr SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN v6_with_irr SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_874 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_876; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_876 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    v4_prefixes _timescaledb_internal.compressed_data,
    v6_prefixes _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN v4_prefixes SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN v6_prefixes SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_876 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_878; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_878 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    advertise_avg _timescaledb_internal.compressed_data,
    advertise_min _timescaledb_internal.compressed_data,
    advertise_max _timescaledb_internal.compressed_data,
    withdraw_avg _timescaledb_internal.compressed_data,
    withdraw_min _timescaledb_internal.compressed_data,
    withdraw_max _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN advertise_avg SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN advertise_min SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN advertise_max SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN withdraw_avg SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN withdraw_min SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN withdraw_max SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_878 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_880; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_880 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    updates _timescaledb_internal.compressed_data,
    withdraws _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN updates SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN withdraws SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_880 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_882; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_882 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    prefix inet,
    prefix_len _timescaledb_internal.compressed_data,
    rd _timescaledb_internal.compressed_data,
    updates _timescaledb_internal.compressed_data,
    withdraws _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN prefix SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN prefix_len SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN rd SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN rd SET STORAGE EXTENDED;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN updates SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN withdraws SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_882 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: _compressed_hypertable_884; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._compressed_hypertable_884 (
    interval_time _timescaledb_internal.compressed_data,
    peer_hash_id uuid,
    rd character varying(128),
    updates _timescaledb_internal.compressed_data,
    withdraws _timescaledb_internal.compressed_data,
    _ts_meta_count integer,
    _ts_meta_sequence_num integer,
    _ts_meta_min_1 timestamp without time zone,
    _ts_meta_max_1 timestamp without time zone
)
WITH (toast_tuple_target='128');
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN interval_time SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN peer_hash_id SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN rd SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN updates SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN withdraws SET STATISTICS 0;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN _ts_meta_count SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN _ts_meta_sequence_num SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN _ts_meta_min_1 SET STATISTICS 1000;
ALTER TABLE ONLY _timescaledb_internal._compressed_hypertable_884 ALTER COLUMN _ts_meta_max_1 SET STATISTICS 1000;


--
-- Name: peer_event_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.peer_event_log (
    id bigint NOT NULL,
    state public.opstate NOT NULL,
    peer_hash_id uuid NOT NULL,
    local_ip inet,
    local_bgp_id inet,
    local_port integer,
    local_hold_time integer,
    geo_ip_start inet,
    local_asn bigint,
    remote_port integer,
    remote_hold_time integer,
    sent_capabilities character varying(4096),
    recv_capabilities character varying(4096),
    bmp_reason smallint,
    bgp_err_code smallint,
    bgp_err_subcode smallint,
    error_text character varying(255),
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: _hyper_860_203_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_860_203_chunk (
    CONSTRAINT constraint_199 CHECK ((("timestamp" >= '2025-12-04 00:00:00'::timestamp without time zone) AND ("timestamp" < '2025-12-11 00:00:00'::timestamp without time zone)))
)
INHERITS (public.peer_event_log);


--
-- Name: ip_rib_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ip_rib_log (
    id bigint NOT NULL,
    base_attr_hash_id uuid NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    peer_hash_id uuid NOT NULL,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    origin_as bigint NOT NULL,
    iswithdrawn boolean NOT NULL
);


--
-- Name: _hyper_862_204_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_862_204_chunk (
    CONSTRAINT constraint_200 CHECK ((("timestamp" >= '2025-12-05 05:00:00'::timestamp without time zone) AND ("timestamp" < '2025-12-05 06:00:00'::timestamp without time zone)))
)
INHERITS (public.ip_rib_log);


--
-- Name: stats_chg_bypeer; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_chg_bypeer (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    updates bigint DEFAULT 0 NOT NULL,
    withdraws bigint DEFAULT 0 NOT NULL
)
WITH (autovacuum_enabled='false');


--
-- Name: _hyper_867_205_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_867_205_chunk (
    CONSTRAINT constraint_201 CHECK (((interval_time >= '2025-12-05 00:00:00'::timestamp without time zone) AND (interval_time < '2025-12-05 06:00:00'::timestamp without time zone)))
)
INHERITS (public.stats_chg_bypeer)
WITH (autovacuum_enabled='false');


--
-- Name: stats_chg_byasn; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_chg_byasn (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    origin_as bigint NOT NULL,
    updates bigint DEFAULT 0 NOT NULL,
    withdraws bigint DEFAULT 0 NOT NULL
)
WITH (autovacuum_enabled='false');


--
-- Name: _hyper_869_207_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_869_207_chunk (
    CONSTRAINT constraint_203 CHECK (((interval_time >= '2025-12-05 00:00:00'::timestamp without time zone) AND (interval_time < '2025-12-05 06:00:00'::timestamp without time zone)))
)
INHERITS (public.stats_chg_byasn)
WITH (autovacuum_enabled='false');


--
-- Name: stats_chg_byprefix; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_chg_byprefix (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    updates bigint DEFAULT 0 NOT NULL,
    withdraws bigint DEFAULT 0 NOT NULL
)
WITH (autovacuum_enabled='false');


--
-- Name: _hyper_871_208_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_871_208_chunk (
    CONSTRAINT constraint_204 CHECK (((interval_time >= '2025-12-05 00:00:00'::timestamp without time zone) AND (interval_time < '2025-12-05 06:00:00'::timestamp without time zone)))
)
INHERITS (public.stats_chg_byprefix)
WITH (autovacuum_enabled='false');


--
-- Name: stats_ip_origins; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_ip_origins (
    id bigint NOT NULL,
    interval_time timestamp(6) without time zone NOT NULL,
    asn bigint NOT NULL,
    v4_prefixes integer DEFAULT 0 NOT NULL,
    v6_prefixes integer DEFAULT 0 NOT NULL,
    v4_with_rpki integer DEFAULT 0 NOT NULL,
    v6_with_rpki integer DEFAULT 0 NOT NULL,
    v4_with_irr integer DEFAULT 0 NOT NULL,
    v6_with_irr integer DEFAULT 0 NOT NULL
);


--
-- Name: _hyper_873_209_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_873_209_chunk (
    CONSTRAINT constraint_205 CHECK (((interval_time >= '2025-11-08 00:00:00'::timestamp without time zone) AND (interval_time < '2025-12-08 00:00:00'::timestamp without time zone)))
)
INHERITS (public.stats_ip_origins);


--
-- Name: stats_peer_rib; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_peer_rib (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    v4_prefixes integer DEFAULT 0 NOT NULL,
    v6_prefixes integer DEFAULT 0 NOT NULL
);


--
-- Name: _hyper_875_206_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_875_206_chunk (
    CONSTRAINT constraint_202 CHECK (((interval_time >= '2025-11-08 00:00:00'::timestamp without time zone) AND (interval_time < '2025-12-08 00:00:00'::timestamp without time zone)))
)
INHERITS (public.stats_peer_rib);


--
-- Name: stats_peer_update_counts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_peer_update_counts (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    advertise_avg integer DEFAULT 0 NOT NULL,
    advertise_min integer DEFAULT 0 NOT NULL,
    advertise_max integer DEFAULT 0 NOT NULL,
    withdraw_avg integer DEFAULT 0 NOT NULL,
    withdraw_min integer DEFAULT 0 NOT NULL,
    withdraw_max integer DEFAULT 0 NOT NULL
);


--
-- Name: _hyper_877_210_chunk; Type: TABLE; Schema: _timescaledb_internal; Owner: -
--

CREATE TABLE _timescaledb_internal._hyper_877_210_chunk (
    CONSTRAINT constraint_206 CHECK (((interval_time >= '2025-11-08 00:00:00'::timestamp without time zone) AND (interval_time < '2025-12-08 00:00:00'::timestamp without time zone)))
)
INHERITS (public.stats_peer_update_counts);


--
-- Name: aggregated_5min; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.aggregated_5min (
    peer_hash_id character varying(255) NOT NULL,
    time_bucket timestamp without time zone NOT NULL,
    update_count integer,
    announcement_count integer,
    withdrawal_count integer,
    unique_prefixes integer
);


--
-- Name: anomaly_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.anomaly_events (
    id integer NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    anomaly_type character varying(50) NOT NULL,
    severity character varying(20) NOT NULL,
    peer_addr inet,
    peer_asn bigint,
    prefix inet,
    description text,
    detector_source character varying(50),
    suggested_actions text[],
    acknowledged boolean DEFAULT false,
    acknowledged_at timestamp with time zone,
    acknowledged_by character varying(100),
    context_data jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: anomaly_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.anomaly_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: anomaly_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.anomaly_events_id_seq OWNED BY public.anomaly_events.id;


--
-- Name: anomaly_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.anomaly_log (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    anomaly_type character varying(50),
    severity character varying(20),
    peer_id character varying(100),
    peer_ip character varying(45),
    peer_asn bigint,
    details jsonb,
    processed boolean DEFAULT false
);


--
-- Name: anomaly_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.anomaly_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: anomaly_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.anomaly_log_id_seq OWNED BY public.anomaly_log.id;


--
-- Name: base_attrs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.base_attrs (
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    origin character varying(16) NOT NULL,
    as_path bigint[] NOT NULL,
    as_path_count smallint DEFAULT 0,
    origin_as bigint,
    next_hop inet,
    med bigint,
    local_pref bigint,
    aggregator character varying(64),
    community_list character varying(15)[],
    ext_community_list character varying(50)[],
    large_community_list character varying(40)[],
    cluster_list character varying(40)[],
    isatomicagg boolean DEFAULT false,
    nexthop_isipv4 boolean DEFAULT true,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    originator_id inet
)
WITH (autovacuum_analyze_threshold='1000', autovacuum_vacuum_threshold='2000', autovacuum_vacuum_cost_limit='200', autovacuum_vacuum_cost_delay='10');


--
-- Name: bgp_communities_applied; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bgp_communities_applied (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    prefix character varying(45),
    peer_ip character varying(45),
    community character varying(50),
    action character varying(20),
    reason character varying(100),
    policy_id integer
);


--
-- Name: bgp_communities_applied_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bgp_communities_applied_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bgp_communities_applied_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bgp_communities_applied_id_seq OWNED BY public.bgp_communities_applied.id;


--
-- Name: bgp_peers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bgp_peers (
    hash_id uuid NOT NULL,
    router_hash_id uuid NOT NULL,
    peer_rd character varying(32) NOT NULL,
    isipv4 boolean DEFAULT true NOT NULL,
    peer_addr inet NOT NULL,
    name character varying(200),
    peer_bgp_id inet,
    peer_as bigint NOT NULL,
    state public.opstate DEFAULT 'down'::public.opstate NOT NULL,
    isl3vpnpeer boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    isprepolicy boolean DEFAULT true,
    geo_ip_start inet,
    local_ip inet,
    local_bgp_id inet,
    local_port integer,
    local_hold_time smallint,
    local_asn bigint,
    remote_port integer,
    remote_hold_time smallint,
    sent_capabilities character varying(4096),
    recv_capabilities character varying(4096),
    bmp_reason smallint,
    bgp_err_code smallint,
    bgp_err_subcode smallint,
    error_text character varying(255),
    islocrib boolean DEFAULT false NOT NULL,
    islocribfiltered boolean DEFAULT false NOT NULL,
    table_name character varying(255),
    region character varying(100),
    tenant_id character varying(100)
);


--
-- Name: bmp_baseline_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bmp_baseline_config (
    id integer NOT NULL,
    config_type character varying(20) NOT NULL,
    tenant_id character varying(100),
    region character varying(100),
    peer_addr inet,
    peer_as bigint,
    baseline_threshold integer DEFAULT 100 NOT NULL,
    announcement_threshold integer,
    withdrawal_threshold integer,
    description text,
    enabled boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    CONSTRAINT bmp_baseline_config_config_type_check CHECK (((config_type)::text = ANY ((ARRAY['global'::character varying, 'tenant'::character varying, 'region'::character varying, 'peer'::character varying])::text[])))
);


--
-- Name: bmp_baseline_config_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bmp_baseline_config_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bmp_baseline_config_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bmp_baseline_config_id_seq OWNED BY public.bmp_baseline_config.id;


--
-- Name: collectors; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.collectors (
    hash_id uuid NOT NULL,
    state public.opstate DEFAULT 'down'::public.opstate,
    admin_id character varying(64) NOT NULL,
    routers character varying(4096),
    router_count smallint DEFAULT 0 NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    name character varying(200),
    ip_address character varying(40)
);


--
-- Name: detector_metrics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.detector_metrics (
    id integer NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    detector_name character varying(50) NOT NULL,
    accuracy numeric(5,4),
    "precision" numeric(5,4),
    recall numeric(5,4),
    f1_score numeric(5,4),
    samples_processed integer,
    avg_processing_time_ms numeric(10,2),
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: detector_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.detector_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: detector_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.detector_metrics_id_seq OWNED BY public.detector_metrics.id;


--
-- Name: geo_ip; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.geo_ip (
    family smallint NOT NULL,
    ip inet NOT NULL,
    country character(2) NOT NULL,
    stateprov character varying(80) NOT NULL,
    city character varying(80) NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    timezone_offset double precision NOT NULL,
    timezone_name character varying(64) NOT NULL,
    isp_name character varying(128) NOT NULL,
    connection_type character varying(64),
    organization_name character varying(128)
);


--
-- Name: global_ip_rib; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.global_ip_rib (
    prefix inet NOT NULL,
    iswithdrawn boolean DEFAULT false NOT NULL,
    prefix_len smallint DEFAULT 0 NOT NULL,
    recv_origin_as bigint NOT NULL,
    rpki_origin_as bigint,
    irr_origin_as bigint,
    irr_source character varying(32),
    irr_descr character varying(255),
    num_peers integer DEFAULT 0,
    advertising_peers integer DEFAULT 0,
    withdrawn_peers integer DEFAULT 0,
    "timestamp" timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    first_added_timestamp timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: hybrid_anomaly_detections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hybrid_anomaly_detections (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    detection_id character varying(100),
    prefix character varying(45),
    prefix_length integer,
    peer_ip character varying(45),
    peer_asn bigint,
    origin_as bigint,
    as_path text,
    next_hop character varying(45),
    event_type character varying(20),
    message_type character varying(50),
    rpki_status character varying(20),
    rpki_anomaly boolean DEFAULT false,
    combined_anomaly boolean DEFAULT false,
    combined_score numeric(5,4),
    combined_severity character varying(20),
    classification character varying(50),
    metadata jsonb
);


--
-- Name: hybrid_anomaly_detections_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hybrid_anomaly_detections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: hybrid_anomaly_detections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hybrid_anomaly_detections_id_seq OWNED BY public.hybrid_anomaly_detections.id;


--
-- Name: info_asn; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.info_asn (
    asn bigint NOT NULL,
    as_name character varying(255),
    org_id character varying(255),
    org_name character varying(255),
    remarks text,
    address character varying(255),
    city character varying(255),
    state_prov character varying(255),
    postal_code character varying(255),
    country character varying(255),
    raw_output text,
    "timestamp" timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    source character varying(64) DEFAULT NULL::character varying
);


--
-- Name: info_route; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.info_route (
    prefix inet NOT NULL,
    prefix_len smallint DEFAULT 0 NOT NULL,
    descr text,
    origin_as bigint NOT NULL,
    source character varying(32) NOT NULL,
    "timestamp" timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ip_rib; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ip_rib (
    hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    peer_hash_id uuid NOT NULL,
    isipv4 boolean NOT NULL,
    origin_as bigint,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    first_added_timestamp timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    iswithdrawn boolean DEFAULT false NOT NULL,
    path_id bigint,
    labels character varying(255),
    isprepolicy boolean DEFAULT true NOT NULL,
    isadjribin boolean DEFAULT true NOT NULL
)
WITH (autovacuum_analyze_threshold='100', autovacuum_vacuum_threshold='200', autovacuum_vacuum_cost_limit='200', autovacuum_vacuum_cost_delay='10');


--
-- Name: ip_rib_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ip_rib_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ip_rib_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ip_rib_log_id_seq OWNED BY public.ip_rib_log.id;


--
-- Name: l3vpn_rib; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.l3vpn_rib (
    hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    peer_hash_id uuid NOT NULL,
    isipv4 boolean NOT NULL,
    rd character varying(128) NOT NULL,
    origin_as bigint,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    first_added_timestamp timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    iswithdrawn boolean DEFAULT false NOT NULL,
    path_id bigint,
    labels character varying(255),
    ext_community_list character varying(50)[],
    isprepolicy boolean DEFAULT true NOT NULL,
    isadjribin boolean DEFAULT true NOT NULL
)
WITH (autovacuum_analyze_threshold='1000', autovacuum_vacuum_threshold='2000', autovacuum_vacuum_cost_limit='200', autovacuum_vacuum_cost_delay='10');


--
-- Name: l3vpn_rib_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.l3vpn_rib_log (
    id bigint NOT NULL,
    base_attr_hash_id uuid,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    rd character varying(128) NOT NULL,
    peer_hash_id uuid NOT NULL,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    origin_as bigint NOT NULL,
    ext_community_list character varying(50)[],
    isprepolicy boolean DEFAULT true NOT NULL,
    isadjribin boolean DEFAULT true NOT NULL,
    iswithdrawn boolean NOT NULL
);


--
-- Name: l3vpn_rib_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.l3vpn_rib_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: l3vpn_rib_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.l3vpn_rib_log_id_seq OWNED BY public.l3vpn_rib_log.id;


--
-- Name: ls_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ls_links (
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    seq bigint DEFAULT 0 NOT NULL,
    mt_id integer DEFAULT 0 NOT NULL,
    interface_addr inet,
    neighbor_addr inet,
    isipv4 boolean DEFAULT true NOT NULL,
    protocol public.ls_proto DEFAULT ''::public.ls_proto,
    local_link_id bigint,
    remote_link_id bigint,
    local_node_hash_id uuid NOT NULL,
    remote_node_hash_id uuid NOT NULL,
    admin_group bigint NOT NULL,
    max_link_bw bigint,
    max_resv_bw bigint,
    unreserved_bw character varying(128),
    te_def_metric bigint,
    protection_type character varying(60),
    mpls_proto_mask public.ls_mpls_proto_mask,
    igp_metric bigint DEFAULT 0 NOT NULL,
    srlg character varying(128),
    name character varying(255),
    local_igp_router_id character varying(46) NOT NULL,
    local_router_id character varying(46) NOT NULL,
    remote_igp_router_id character varying(46) NOT NULL,
    remote_router_id character varying(46) NOT NULL,
    local_asn bigint DEFAULT 0 NOT NULL,
    remote_asn bigint DEFAULT 0 NOT NULL,
    peer_node_sid character varying(128),
    sr_adjacency_sids character varying(255),
    iswithdrawn boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ls_links_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ls_links_log (
    id bigint NOT NULL,
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    seq bigint DEFAULT 0 NOT NULL,
    mt_id integer DEFAULT 0 NOT NULL,
    interface_addr inet,
    neighbor_addr inet,
    isipv4 boolean DEFAULT true NOT NULL,
    protocol public.ls_proto DEFAULT ''::public.ls_proto,
    local_link_id bigint,
    remote_link_id bigint,
    local_node_hash_id uuid NOT NULL,
    remote_node_hash_id uuid NOT NULL,
    admin_group bigint NOT NULL,
    max_link_bw bigint,
    max_resv_bw bigint,
    unreserved_bw character varying(128),
    te_def_metric bigint,
    protection_type character varying(60),
    mpls_proto_mask public.ls_mpls_proto_mask,
    igp_metric bigint DEFAULT 0 NOT NULL,
    srlg character varying(128),
    name character varying(255),
    local_igp_router_id character varying(46) NOT NULL,
    local_router_id character varying(46) NOT NULL,
    remote_igp_router_id character varying(46) NOT NULL,
    remote_router_id character varying(46) NOT NULL,
    local_asn bigint DEFAULT 0 NOT NULL,
    remote_asn bigint DEFAULT 0 NOT NULL,
    peer_node_sid character varying(128),
    sr_adjacency_sids character varying(255),
    iswithdrawn boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ls_links_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ls_links_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ls_links_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ls_links_log_id_seq OWNED BY public.ls_links_log.id;


--
-- Name: ls_nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ls_nodes (
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    seq bigint DEFAULT 0 NOT NULL,
    asn bigint NOT NULL,
    bgp_ls_id bigint DEFAULT 0 NOT NULL,
    igp_router_id character varying(46) NOT NULL,
    ospf_area_id character varying(16) NOT NULL,
    protocol public.ls_proto DEFAULT ''::public.ls_proto,
    router_id character varying(46) NOT NULL,
    isis_area_id character varying(46),
    flags character varying(20),
    name character varying(255),
    mt_ids character varying(128),
    sr_capabilities character varying(255),
    iswithdrawn boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ls_nodes_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ls_nodes_log (
    id bigint NOT NULL,
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    seq bigint DEFAULT 0 NOT NULL,
    asn bigint NOT NULL,
    bgp_ls_id bigint DEFAULT 0 NOT NULL,
    igp_router_id character varying(46) NOT NULL,
    ospf_area_id character varying(16) NOT NULL,
    protocol public.ls_proto DEFAULT ''::public.ls_proto,
    router_id character varying(46) NOT NULL,
    isis_area_id character varying(46),
    flags character varying(20),
    name character varying(255),
    mt_ids character varying(128),
    sr_capabilities character varying(255),
    iswithdrawn boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ls_nodes_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ls_nodes_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ls_nodes_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ls_nodes_log_id_seq OWNED BY public.ls_nodes_log.id;


--
-- Name: ls_prefixes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ls_prefixes (
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    seq bigint DEFAULT 0 NOT NULL,
    local_node_hash_id uuid NOT NULL,
    mt_id integer DEFAULT 0 NOT NULL,
    protocol public.ls_proto DEFAULT ''::public.ls_proto,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    ospf_route_type public.ospf_route_type DEFAULT ''::public.ospf_route_type NOT NULL,
    igp_flags character varying(20),
    isipv4 boolean DEFAULT true NOT NULL,
    route_tag bigint DEFAULT 0 NOT NULL,
    ext_route_tag bigint DEFAULT 0 NOT NULL,
    metric bigint DEFAULT 0 NOT NULL,
    ospf_fwd_addr inet,
    sr_prefix_sids character varying(255),
    iswithdrawn boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ls_prefixes_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ls_prefixes_log (
    id bigint NOT NULL,
    hash_id uuid NOT NULL,
    peer_hash_id uuid NOT NULL,
    base_attr_hash_id uuid,
    seq bigint DEFAULT 0 NOT NULL,
    local_node_hash_id uuid NOT NULL,
    mt_id integer DEFAULT 0 NOT NULL,
    protocol public.ls_proto DEFAULT ''::public.ls_proto,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    ospf_route_type public.ospf_route_type DEFAULT ''::public.ospf_route_type NOT NULL,
    igp_flags character varying(20),
    isipv4 boolean DEFAULT true NOT NULL,
    route_tag bigint DEFAULT 0 NOT NULL,
    ext_route_tag bigint DEFAULT 0 NOT NULL,
    metric bigint DEFAULT 0 NOT NULL,
    ospf_fwd_addr inet,
    sr_prefix_sids character varying(255),
    iswithdrawn boolean DEFAULT false NOT NULL,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: ls_prefixes_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ls_prefixes_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ls_prefixes_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ls_prefixes_log_id_seq OWNED BY public.ls_prefixes_log.id;


--
-- Name: pdb_exchange_peers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pdb_exchange_peers (
    ix_id integer NOT NULL,
    ix_name character varying(128) NOT NULL,
    ix_prefix_v4 inet,
    ix_prefix_v6 inet,
    ix_country character varying(12),
    ix_city character varying(128),
    ix_region character varying(128),
    rs_peer boolean DEFAULT false NOT NULL,
    peer_name character varying(255) NOT NULL,
    peer_ipv4 inet DEFAULT '0.0.0.0'::inet NOT NULL,
    peer_ipv6 inet DEFAULT '::'::inet NOT NULL,
    peer_asn bigint NOT NULL,
    speed integer,
    policy character varying(64),
    poc_policy_email character varying(255),
    poc_noc_email character varying(255),
    "timestamp" timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: peer_event_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.peer_event_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: peer_event_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.peer_event_log_id_seq OWNED BY public.peer_event_log.id;


--
-- Name: peer_severity_states; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.peer_severity_states (
    peer_hash_id character varying(255) NOT NULL,
    peer_addr inet,
    peer_asn bigint,
    current_severity character varying(20),
    previous_severity character varying(20),
    consecutive_windows integer,
    cooldown_until timestamp without time zone,
    baseline_rate integer,
    current_rate integer,
    last_updated timestamp without time zone
);


--
-- Name: peer_state_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.peer_state_history (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    peer_id character varying(100),
    peer_ip character varying(45),
    peer_asn bigint,
    old_state character varying(20),
    new_state character varying(20),
    uptime_seconds integer,
    anomaly_detected boolean DEFAULT false,
    anomaly_type character varying(50),
    metadata jsonb
);


--
-- Name: peer_state_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.peer_state_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: peer_state_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.peer_state_history_id_seq OWNED BY public.peer_state_history.id;


--
-- Name: policy_execution_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.policy_execution_log (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    recommendation_id integer,
    action_type character varying(50),
    action_parameters jsonb,
    execution_status character varying(50),
    execution_result jsonb,
    rfc_reference character varying(100),
    operator_approved boolean DEFAULT false,
    operator_id character varying(100),
    error_message text
);


--
-- Name: policy_execution_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.policy_execution_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: policy_execution_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.policy_execution_log_id_seq OWNED BY public.policy_execution_log.id;


--
-- Name: policy_recommendations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.policy_recommendations (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    detection_id character varying(100),
    anomaly_type character varying(50),
    policy_severity character varying(20),
    recommended_actions jsonb,
    auto_remediation boolean DEFAULT false,
    requires_approval boolean DEFAULT false,
    escalation_timeout integer,
    rfc_compliance text[],
    policy_executed boolean DEFAULT false,
    execution_timestamp timestamp without time zone,
    execution_status character varying(50)
);


--
-- Name: policy_recommendations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.policy_recommendations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: policy_recommendations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.policy_recommendations_id_seq OWNED BY public.policy_recommendations.id;


--
-- Name: route_dampening_state; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.route_dampening_state (
    id integer NOT NULL,
    prefix character varying(45),
    penalty numeric(10,2),
    suppressed boolean DEFAULT false,
    flap_count integer DEFAULT 0,
    last_update timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    suppress_time timestamp without time zone,
    reuse_eligible_time timestamp without time zone
);


--
-- Name: route_dampening_state_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.route_dampening_state_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: route_dampening_state_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.route_dampening_state_id_seq OWNED BY public.route_dampening_state.id;


--
-- Name: routers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.routers (
    hash_id uuid NOT NULL,
    name character varying(200) NOT NULL,
    ip_address inet NOT NULL,
    router_as bigint,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    description character varying(255),
    state public.opstate DEFAULT 'down'::public.opstate,
    ispassive boolean DEFAULT false,
    term_reason_code integer,
    term_reason_text character varying(255),
    term_data text,
    init_data text,
    geo_ip_start inet,
    collector_hash_id uuid NOT NULL,
    bgp_id inet
);


--
-- Name: rpki_validator; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rpki_validator (
    prefix inet NOT NULL,
    prefix_len smallint DEFAULT 0 NOT NULL,
    prefix_len_max smallint DEFAULT 0 NOT NULL,
    origin_as bigint NOT NULL,
    "timestamp" timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: stat_reports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stat_reports (
    id bigint NOT NULL,
    peer_hash_id uuid NOT NULL,
    prefixes_rejected bigint,
    known_dup_prefixes bigint,
    known_dup_withdraws bigint,
    updates_invalid_by_cluster_list bigint,
    updates_invalid_by_as_path_loop bigint,
    updates_invalid_by_originagtor_id bigint,
    updates_invalid_by_as_confed_loop bigint,
    num_routes_adj_rib_in bigint,
    num_routes_local_rib bigint,
    "timestamp" timestamp(6) without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: stat_reports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.stat_reports_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: stat_reports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.stat_reports_id_seq OWNED BY public.stat_reports.id;


--
-- Name: stats_ip_origins_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.stats_ip_origins_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: stats_ip_origins_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.stats_ip_origins_id_seq OWNED BY public.stats_ip_origins.id;


--
-- Name: stats_l3vpn_chg_bypeer; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_l3vpn_chg_bypeer (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    updates bigint DEFAULT 0 NOT NULL,
    withdraws bigint DEFAULT 0 NOT NULL
);


--
-- Name: stats_l3vpn_chg_byprefix; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_l3vpn_chg_byprefix (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    prefix inet NOT NULL,
    prefix_len smallint NOT NULL,
    rd character varying(128) NOT NULL,
    updates bigint DEFAULT 0 NOT NULL,
    withdraws bigint DEFAULT 0 NOT NULL
);


--
-- Name: stats_l3vpn_chg_byrd; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stats_l3vpn_chg_byrd (
    interval_time timestamp(6) without time zone NOT NULL,
    peer_hash_id uuid NOT NULL,
    rd character varying(128) NOT NULL,
    updates bigint DEFAULT 0 NOT NULL,
    withdraws bigint DEFAULT 0 NOT NULL
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    username character varying(50) NOT NULL,
    password character varying(50) NOT NULL,
    type public.user_role NOT NULL
);


--
-- Name: v_anomaly_type_summary; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.v_anomaly_type_summary AS
 SELECT anomaly_events.anomaly_type,
    count(*) AS event_count,
    max((anomaly_events.severity)::text) AS max_severity,
    array_agg(DISTINCT (anomaly_events.peer_addr)::text) AS affected_peers,
    min(anomaly_events."timestamp") AS first_occurrence,
    max(anomaly_events."timestamp") AS last_occurrence
   FROM public.anomaly_events
  WHERE (anomaly_events."timestamp" >= (now() - '7 days'::interval))
  GROUP BY anomaly_events.anomaly_type
  WITH NO DATA;


--
-- Name: v_baseline_summary; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_baseline_summary AS
 SELECT bmp_baseline_config.config_type,
    COALESCE(bmp_baseline_config.tenant_id, '-'::character varying) AS tenant,
    COALESCE(bmp_baseline_config.region, '-'::character varying) AS region,
    COALESCE((bmp_baseline_config.peer_addr)::text, '-'::text) AS peer_ip,
    COALESCE((bmp_baseline_config.peer_as)::text, '-'::text) AS peer_as,
    bmp_baseline_config.baseline_threshold,
    bmp_baseline_config.announcement_threshold,
    bmp_baseline_config.withdrawal_threshold,
    bmp_baseline_config.description,
    bmp_baseline_config.enabled
   FROM public.bmp_baseline_config
  ORDER BY
        CASE bmp_baseline_config.config_type
            WHEN 'peer'::text THEN 1
            WHEN 'region'::text THEN 2
            WHEN 'tenant'::text THEN 3
            WHEN 'global'::text THEN 4
            ELSE NULL::integer
        END;


--
-- Name: v_ip_routes; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_ip_routes AS
 SELECT
        CASE
            WHEN (length((rtr.name)::text) > 0) THEN (rtr.name)::text
            ELSE host(rtr.ip_address)
        END AS routername,
        CASE
            WHEN (length((p.name)::text) > 0) THEN (p.name)::text
            ELSE host(p.peer_addr)
        END AS peername,
    r.prefix,
    r.prefix_len AS prefixlen,
    attr.origin,
    r.origin_as,
    attr.med,
    attr.local_pref AS localpref,
    attr.next_hop AS nh,
    attr.as_path,
    attr.as_path_count AS aspath_count,
    attr.community_list AS communities,
    attr.ext_community_list AS extcommunities,
    attr.large_community_list AS largecommunities,
    attr.cluster_list AS clusterlist,
    attr.originator_id AS originator,
    attr.aggregator,
    p.peer_addr AS peeraddress,
    p.peer_as AS peerasn,
    r.isipv4,
    p.isipv4 AS ispeeripv4,
    p.isl3vpnpeer AS ispeervpn,
    r."timestamp" AS lastmodified,
    r.first_added_timestamp AS firstaddedtimestamp,
    r.path_id,
    r.labels,
    r.hash_id AS rib_hash_id,
    r.base_attr_hash_id AS base_hash_id,
    r.peer_hash_id,
    rtr.hash_id AS router_hash_id,
    r.iswithdrawn,
    r.isprepolicy,
    r.isadjribin
   FROM (((public.ip_rib r
     JOIN public.bgp_peers p ON ((r.peer_hash_id = p.hash_id)))
     JOIN public.base_attrs attr ON (((attr.hash_id = r.base_attr_hash_id) AND (attr.peer_hash_id = r.peer_hash_id))))
     JOIN public.routers rtr ON ((p.router_hash_id = rtr.hash_id)));


--
-- Name: v_ip_routes_geo; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_ip_routes_geo AS
 SELECT
        CASE
            WHEN (length((rtr.name)::text) > 0) THEN (rtr.name)::text
            ELSE host(rtr.ip_address)
        END AS routername,
        CASE
            WHEN (length((p.name)::text) > 0) THEN (p.name)::text
            ELSE host(p.peer_addr)
        END AS peername,
    r.prefix,
    r.prefix_len AS prefixlen,
    attr.origin,
    r.origin_as,
    attr.med,
    attr.local_pref AS localpref,
    attr.next_hop AS nh,
    attr.as_path,
    attr.as_path_count AS aspath_count,
    attr.community_list AS communities,
    attr.ext_community_list AS extcommunities,
    attr.large_community_list AS largecommunities,
    attr.cluster_list AS clusterlist,
    attr.originator_id AS originator,
    attr.aggregator,
    p.peer_addr AS peeraddress,
    p.peer_as AS peerasn,
    r.isipv4,
    p.isipv4 AS ispeeripv4,
    p.isl3vpnpeer AS ispeervpn,
    r."timestamp" AS lastmodified,
    r.first_added_timestamp AS firstaddedtimestamp,
    r.path_id,
    r.labels,
    r.hash_id AS rib_hash_id,
    r.base_attr_hash_id AS base_hash_id,
    r.peer_hash_id,
    rtr.hash_id AS router_hash_id,
    r.iswithdrawn,
    r.isprepolicy,
    r.isadjribin,
    g.ip AS geo_ip,
    g.city,
    g.stateprov,
    g.country,
    g.latitude,
    g.longitude
   FROM ((((public.ip_rib r
     JOIN public.bgp_peers p ON ((r.peer_hash_id = p.hash_id)))
     JOIN public.base_attrs attr ON (((attr.hash_id = r.base_attr_hash_id) AND (attr.peer_hash_id = r.peer_hash_id))))
     JOIN public.routers rtr ON ((p.router_hash_id = rtr.hash_id)))
     LEFT JOIN public.geo_ip g ON ((g.ip && (host(r.prefix))::inet)))
  WHERE (r.iswithdrawn = false);


--
-- Name: v_ip_routes_history; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_ip_routes_history AS
 SELECT
        CASE
            WHEN (length((rtr.name)::text) > 0) THEN (rtr.name)::text
            ELSE host(rtr.ip_address)
        END AS routername,
    rtr.ip_address AS routeraddress,
        CASE
            WHEN (length((p.name)::text) > 0) THEN (p.name)::text
            ELSE host(p.peer_addr)
        END AS peername,
    log.prefix,
    log.prefix_len AS prefixlen,
    attr.origin,
    log.origin_as,
    attr.med,
    attr.local_pref AS localpref,
    attr.next_hop AS nh,
    attr.as_path,
    attr.as_path_count AS aspath_count,
    attr.community_list AS communities,
    attr.ext_community_list AS extcommunities,
    attr.large_community_list AS largecommunities,
    attr.cluster_list AS clusterlist,
    attr.originator_id AS originator,
    attr.aggregator,
    p.peer_addr AS peerip,
    p.peer_as AS peerasn,
    p.isipv4 AS ispeeripv4,
    p.isl3vpnpeer AS ispeervpn,
    log.id,
    log."timestamp" AS lastmodified,
        CASE
            WHEN log.iswithdrawn THEN 'Withdrawn'::text
            ELSE 'Advertised'::text
        END AS event,
    log.base_attr_hash_id,
    log.peer_hash_id,
    rtr.hash_id AS router_hash_id
   FROM (((public.ip_rib_log log
     JOIN public.base_attrs attr ON (((log.base_attr_hash_id = attr.hash_id) AND (log.peer_hash_id = attr.peer_hash_id))))
     JOIN public.bgp_peers p ON ((log.peer_hash_id = p.hash_id)))
     JOIN public.routers rtr ON ((p.router_hash_id = rtr.hash_id)));


--
-- Name: v_l3vpn_routes; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_l3vpn_routes AS
 SELECT
        CASE
            WHEN (length((rtr.name)::text) > 0) THEN (rtr.name)::text
            ELSE host(rtr.ip_address)
        END AS routername,
        CASE
            WHEN (length((p.name)::text) > 0) THEN (p.name)::text
            ELSE host(p.peer_addr)
        END AS peername,
    r.rd,
    r.prefix,
    r.prefix_len AS prefixlen,
    attr.origin,
    r.origin_as,
    attr.med,
    attr.local_pref AS localpref,
    attr.next_hop AS nh,
    attr.as_path,
    attr.as_path_count AS aspath_count,
    attr.community_list AS communities,
    r.ext_community_list AS extcommunities,
    attr.large_community_list AS largecommunities,
    attr.cluster_list AS clusterlist,
    attr.aggregator,
    p.peer_addr AS peeraddress,
    p.peer_as AS peerasn,
    r.isipv4,
    p.isipv4 AS ispeeripv4,
    p.isl3vpnpeer AS ispeervpn,
    r."timestamp" AS lastmodified,
    r.first_added_timestamp AS firstaddedtimestamp,
    r.path_id,
    r.labels,
    r.hash_id AS rib_hash_id,
    r.base_attr_hash_id AS base_hash_id,
    r.peer_hash_id,
    rtr.hash_id AS router_hash_id,
    r.iswithdrawn,
    r.isprepolicy,
    r.isadjribin
   FROM (((public.l3vpn_rib r
     JOIN public.bgp_peers p ON ((r.peer_hash_id = p.hash_id)))
     JOIN public.base_attrs attr ON (((attr.hash_id = r.base_attr_hash_id) AND (attr.peer_hash_id = r.peer_hash_id))))
     JOIN public.routers rtr ON ((p.router_hash_id = rtr.hash_id)));


--
-- Name: v_l3vpn_routes_history; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_l3vpn_routes_history AS
 SELECT r.id,
        CASE
            WHEN (length((rtr.name)::text) > 0) THEN (rtr.name)::text
            ELSE host(rtr.ip_address)
        END AS routername,
        CASE
            WHEN (length((p.name)::text) > 0) THEN (p.name)::text
            ELSE host(p.peer_addr)
        END AS peername,
    r.rd,
    r.prefix,
    r.prefix_len AS prefixlen,
    attr.origin,
    r.origin_as,
    attr.med,
    attr.local_pref AS localpref,
    attr.next_hop AS nh,
    attr.as_path,
    attr.as_path_count AS aspath_count,
    attr.community_list AS communities,
    r.ext_community_list AS extcommunities,
    attr.large_community_list AS largecommunities,
    attr.cluster_list AS clusterlist,
    attr.aggregator,
    p.peer_addr AS peeraddress,
    p.peer_as AS peerasn,
    p.isipv4 AS ispeeripv4,
    p.isl3vpnpeer AS ispeervpn,
    r."timestamp" AS lastmodified,
    r.base_attr_hash_id AS base_hash_id,
    r.peer_hash_id,
    rtr.hash_id AS router_hash_id,
        CASE
            WHEN r.iswithdrawn THEN 'Withdrawn'::text
            ELSE 'Advertised'::text
        END AS event,
    r.isprepolicy,
    r.isadjribin
   FROM (((public.l3vpn_rib_log r
     JOIN public.bgp_peers p ON ((r.peer_hash_id = p.hash_id)))
     JOIN public.base_attrs attr ON (((attr.hash_id = r.base_attr_hash_id) AND (attr.peer_hash_id = r.peer_hash_id))))
     JOIN public.routers rtr ON ((p.router_hash_id = rtr.hash_id)));


--
-- Name: v_ls_links; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_ls_links AS
 SELECT localn.name AS local_router_name,
    remoten.name AS remote_router_name,
    localn.igp_router_id AS local_igp_routerid,
    localn.router_id AS local_routerid,
    remoten.igp_router_id AS remote_igp_routerid,
    remoten.router_id AS remote_routerid,
    localn.seq,
    localn.bgp_ls_id AS bgpls_id,
        CASE
            WHEN (ln.protocol = ANY (ARRAY['OSPFv2'::public.ls_proto, 'OSPFv3'::public.ls_proto])) THEN localn.ospf_area_id
            ELSE localn.isis_area_id
        END AS areaid,
    ln.mt_id,
    ln.interface_addr AS interfaceip,
    ln.neighbor_addr AS neighborip,
    ln.isipv4,
    ln.protocol,
    ln.igp_metric,
    ln.local_link_id,
    ln.remote_link_id,
    ln.admin_group,
    ln.max_link_bw,
    ln.max_resv_bw,
    ln.unreserved_bw,
    ln.te_def_metric,
    ln.mpls_proto_mask,
    ln.srlg,
    ln.name,
    ln."timestamp",
    ln.local_node_hash_id,
    ln.remote_node_hash_id,
    localn.igp_router_id AS localn_igp_router_id,
    remoten.igp_router_id AS remoten_igp_router_id,
    ln.base_attr_hash_id,
    ln.peer_hash_id,
        CASE
            WHEN ln.iswithdrawn THEN 'WITHDRAWN'::text
            ELSE 'ACTIVE'::text
        END AS state
   FROM ((public.ls_links ln
     JOIN public.ls_nodes localn ON (((ln.local_node_hash_id = localn.hash_id) AND (ln.peer_hash_id = localn.peer_hash_id))))
     JOIN public.ls_nodes remoten ON (((ln.remote_node_hash_id = remoten.hash_id) AND (ln.peer_hash_id = remoten.peer_hash_id))));


--
-- Name: v_ls_nodes; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_ls_nodes AS
 SELECT r.name AS routername,
    r.ip_address AS routerip,
    p.name AS peername,
    p.peer_addr AS peerip,
    ls_nodes.igp_router_id AS igp_routerid,
    ls_nodes.name AS nodename,
        CASE
            WHEN ls_nodes.iswithdrawn THEN 'WITHDRAWN'::text
            ELSE 'ACTIVE'::text
        END AS state,
        CASE
            WHEN (ls_nodes.protocol = ANY (ARRAY['OSPFv2'::public.ls_proto, 'OSPFv3'::public.ls_proto])) THEN ls_nodes.router_id
            ELSE ls_nodes.igp_router_id
        END AS routerid,
    ls_nodes.seq,
    ls_nodes.bgp_ls_id AS bgpls_id,
    ls_nodes.ospf_area_id AS ospfareaid,
    ls_nodes.isis_area_id AS isisareaid,
    ls_nodes.protocol,
    ls_nodes.flags,
    ls_nodes."timestamp",
    ls_nodes.asn,
    base_attrs.as_path,
    base_attrs.local_pref AS localpref,
    base_attrs.med,
    base_attrs.next_hop AS nh,
    ls_nodes.mt_ids,
    ls_nodes.hash_id,
    ls_nodes.base_attr_hash_id,
    ls_nodes.peer_hash_id,
    r.hash_id AS router_hash_id
   FROM (((public.ls_nodes
     LEFT JOIN public.base_attrs ON (((ls_nodes.base_attr_hash_id = base_attrs.hash_id) AND (ls_nodes.peer_hash_id = base_attrs.peer_hash_id))))
     JOIN public.bgp_peers p ON ((p.hash_id = ls_nodes.peer_hash_id)))
     JOIN public.routers r ON ((p.router_hash_id = r.hash_id)))
  WHERE ((NOT ((ls_nodes.igp_router_id)::text ~ '\..[1-9A-F]00$'::text)) AND ((ls_nodes.igp_router_id)::text !~~ '%]'::text));


--
-- Name: v_ls_prefixes; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_ls_prefixes AS
 SELECT localn.name AS local_router_name,
    localn.igp_router_id AS local_igp_routerid,
    localn.router_id AS local_routerid,
    lp.seq,
    lp.mt_id,
    lp.prefix,
    lp.prefix_len,
    lp.ospf_route_type,
    lp.metric,
    lp.protocol,
    lp."timestamp",
    lp.peer_hash_id,
    lp.local_node_hash_id,
        CASE
            WHEN lp.iswithdrawn THEN 'WITHDRAWN'::text
            ELSE 'ACTIVE'::text
        END AS state
   FROM (public.ls_prefixes lp
     JOIN public.ls_nodes localn ON (((localn.peer_hash_id = lp.peer_hash_id) AND (lp.local_node_hash_id = localn.hash_id))));


--
-- Name: v_peers; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_peers AS
 SELECT
        CASE
            WHEN (length((rtr.name)::text) > 0) THEN (rtr.name)::text
            ELSE host(rtr.ip_address)
        END AS routername,
    rtr.ip_address AS routerip,
    p.local_ip AS localip,
    p.local_port AS localport,
    p.local_asn AS localasn,
    p.local_bgp_id AS localbgpid,
        CASE
            WHEN (length((p.name)::text) > 0) THEN (p.name)::text
            ELSE host(p.peer_addr)
        END AS peername,
    p.peer_addr AS peerip,
    p.remote_port AS peerport,
    p.peer_as AS peerasn,
    p.peer_bgp_id AS peerbgpid,
    p.local_hold_time AS localholdtime,
    p.remote_hold_time AS peerholdtime,
    p.state AS peer_state,
    rtr.state AS router_state,
    p.isipv4 AS ispeeripv4,
    p.isl3vpnpeer AS ispeervpn,
    p.isprepolicy,
    p."timestamp" AS lastmodified,
    p.bmp_reason AS lastbmpreasoncode,
    p.bgp_err_code AS lastdowncode,
    p.bgp_err_subcode AS lastdownsubcode,
    p.error_text AS lastdownmessage,
    p."timestamp" AS lastdowntimestamp,
    p.sent_capabilities AS sentcapabilities,
    p.recv_capabilities AS recvcapabilities,
    w.as_name,
    p.islocrib,
    p.islocribfiltered,
    p.table_name,
    p.hash_id AS peer_hash_id,
    rtr.hash_id AS router_hash_id,
    p.geo_ip_start
   FROM ((public.bgp_peers p
     JOIN public.routers rtr ON ((p.router_hash_id = rtr.hash_id)))
     LEFT JOIN public.info_asn w ON ((p.peer_as = w.asn)));


--
-- Name: v_severity_heatmap; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_severity_heatmap AS
 SELECT date_trunc('hour'::text, peer_severity_states.last_updated) AS time_bucket,
    peer_severity_states.current_severity AS severity,
    count(DISTINCT peer_severity_states.peer_hash_id) AS peer_count
   FROM public.peer_severity_states
  WHERE (peer_severity_states.last_updated >= (now() - '24:00:00'::interval))
  GROUP BY (date_trunc('hour'::text, peer_severity_states.last_updated)), peer_severity_states.current_severity
  ORDER BY (date_trunc('hour'::text, peer_severity_states.last_updated)) DESC, peer_severity_states.current_severity;


--
-- Name: _hyper_860_203_chunk id; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_860_203_chunk ALTER COLUMN id SET DEFAULT nextval('public.peer_event_log_id_seq'::regclass);


--
-- Name: _hyper_860_203_chunk timestamp; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_860_203_chunk ALTER COLUMN "timestamp" SET DEFAULT (now() AT TIME ZONE 'utc'::text);


--
-- Name: _hyper_862_204_chunk id; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_862_204_chunk ALTER COLUMN id SET DEFAULT nextval('public.ip_rib_log_id_seq'::regclass);


--
-- Name: _hyper_862_204_chunk timestamp; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_862_204_chunk ALTER COLUMN "timestamp" SET DEFAULT (now() AT TIME ZONE 'utc'::text);


--
-- Name: _hyper_867_205_chunk updates; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_867_205_chunk ALTER COLUMN updates SET DEFAULT 0;


--
-- Name: _hyper_867_205_chunk withdraws; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_867_205_chunk ALTER COLUMN withdraws SET DEFAULT 0;


--
-- Name: _hyper_869_207_chunk updates; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_869_207_chunk ALTER COLUMN updates SET DEFAULT 0;


--
-- Name: _hyper_869_207_chunk withdraws; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_869_207_chunk ALTER COLUMN withdraws SET DEFAULT 0;


--
-- Name: _hyper_871_208_chunk updates; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_871_208_chunk ALTER COLUMN updates SET DEFAULT 0;


--
-- Name: _hyper_871_208_chunk withdraws; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_871_208_chunk ALTER COLUMN withdraws SET DEFAULT 0;


--
-- Name: _hyper_873_209_chunk id; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN id SET DEFAULT nextval('public.stats_ip_origins_id_seq'::regclass);


--
-- Name: _hyper_873_209_chunk v4_prefixes; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN v4_prefixes SET DEFAULT 0;


--
-- Name: _hyper_873_209_chunk v6_prefixes; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN v6_prefixes SET DEFAULT 0;


--
-- Name: _hyper_873_209_chunk v4_with_rpki; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN v4_with_rpki SET DEFAULT 0;


--
-- Name: _hyper_873_209_chunk v6_with_rpki; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN v6_with_rpki SET DEFAULT 0;


--
-- Name: _hyper_873_209_chunk v4_with_irr; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN v4_with_irr SET DEFAULT 0;


--
-- Name: _hyper_873_209_chunk v6_with_irr; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_873_209_chunk ALTER COLUMN v6_with_irr SET DEFAULT 0;


--
-- Name: _hyper_875_206_chunk v4_prefixes; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_875_206_chunk ALTER COLUMN v4_prefixes SET DEFAULT 0;


--
-- Name: _hyper_875_206_chunk v6_prefixes; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_875_206_chunk ALTER COLUMN v6_prefixes SET DEFAULT 0;


--
-- Name: _hyper_877_210_chunk advertise_avg; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_877_210_chunk ALTER COLUMN advertise_avg SET DEFAULT 0;


--
-- Name: _hyper_877_210_chunk advertise_min; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_877_210_chunk ALTER COLUMN advertise_min SET DEFAULT 0;


--
-- Name: _hyper_877_210_chunk advertise_max; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_877_210_chunk ALTER COLUMN advertise_max SET DEFAULT 0;


--
-- Name: _hyper_877_210_chunk withdraw_avg; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_877_210_chunk ALTER COLUMN withdraw_avg SET DEFAULT 0;


--
-- Name: _hyper_877_210_chunk withdraw_min; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_877_210_chunk ALTER COLUMN withdraw_min SET DEFAULT 0;


--
-- Name: _hyper_877_210_chunk withdraw_max; Type: DEFAULT; Schema: _timescaledb_internal; Owner: -
--

ALTER TABLE ONLY _timescaledb_internal._hyper_877_210_chunk ALTER COLUMN withdraw_max SET DEFAULT 0;


--
-- Name: anomaly_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.anomaly_events ALTER COLUMN id SET DEFAULT nextval('public.anomaly_events_id_seq'::regclass);


--
-- Name: anomaly_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.anomaly_log ALTER COLUMN id SET DEFAULT nextval('public.anomaly_log_id_seq'::regclass);


--
-- Name: bgp_communities_applied id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bgp_communities_applied ALTER COLUMN id SET DEFAULT nextval('public.bgp_communities_applied_id_seq'::regclass);


--
-- Name: bmp_baseline_config id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bmp_baseline_config ALTER COLUMN id SET DEFAULT nextval('public.bmp_baseline_config_id_seq'::regclass);


--
-- Name: detector_metrics id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detector_metrics ALTER COLUMN id SET DEFAULT nextval('public.detector_metrics_id_seq'::regclass);


--
-- Name: hybrid_anomaly_detections id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hybrid_anomaly_detections ALTER COLUMN id SET DEFAULT nextval('public.hybrid_anomaly_detections_id_seq'::regclass);


--
-- Name: ip_rib_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ip_rib_log ALTER COLUMN id SET DEFAULT nextval('public.ip_rib_log_id_seq'::regclass);


--
-- Name: l3vpn_rib_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.l3vpn_rib_log ALTER COLUMN id SET DEFAULT nextval('public.l3vpn_rib_log_id_seq'::regclass);


--
-- Name: ls_links_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ls_links_log ALTER COLUMN id SET DEFAULT nextval('public.ls_links_log_id_seq'::regclass);


--
-- Name: ls_nodes_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ls_nodes_log ALTER COLUMN id SET DEFAULT nextval('public.ls_nodes_log_id_seq'::regclass);


--
-- Name: ls_prefixes_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ls_prefixes_log ALTER COLUMN id SET DEFAULT nextval('public.ls_prefixes_log_id_seq'::regclass);


--
-- Name: peer_event_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.peer_event_log ALTER COLUMN id SET DEFAULT nextval('public.peer_event_log_id_seq'::regclass);


--
-- Name: peer_state_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.peer_state_history ALTER COLUMN id SET DEFAULT nextval('public.peer_state_history_id_seq'::regclass);


--
-- Name: policy_execution_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.policy_execution_log ALTER COLUMN id SET DEFAULT nextval('public.policy_execution_log_id_seq'::regclass);


--
-- Name: policy_recommendations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.policy_recommendations ALTER COLUMN id SET DEFAULT nextval('public.policy_recommendations_id_seq'::regclass);


--
-- Name: route_dampening_state id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.route_dampening_state ALTER COLUMN id SET DEFAULT nextval('public.route_dampening_state_id_seq'::regclass);


--
-- Name: stat_reports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stat_reports ALTER COLUMN id SET DEFAULT nextval('public.stat_reports_id_seq'::regclass);


--
-- Name: stats_ip_origins id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stats_ip_origins ALTER COLUMN id SET DEFAULT nextval('public.stats_ip_origins_id_seq'::regclass);


--
-- PostgreSQL database dump complete
--
-- Table for Raw BGP Data (Input for ML)
CREATE TABLE raw_bgp_data (
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    peer_addr TEXT,
    announcements INTEGER,
    withdrawals INTEGER,
    total_updates INTEGER,
    withdrawal_ratio REAL,
    flap_count INTEGER,
    path_length REAL,
    unique_peers INTEGER,
    message_rate REAL,
    session_resets INTEGER
);

-- Convert to TimescaleDB Hypertable
SELECT create_hypertable('raw_bgp_data', 'timestamp', if_not_exists => TRUE);

-- Table for ML Results (Output from ML)
CREATE TABLE anomaly_alerts (
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    peer_addr TEXT,
    ensemble_score REAL NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL,
    lstm_error REAL,
    if_score REAL,
    raw_data_id BIGINT  -- Link to raw data if needed, or use timestamp as key
);

-- Convert to TimescaleDB Hypertable
SELECT create_hypertable('anomaly_alerts', 'timestamp', if_not_exists => TRUE);