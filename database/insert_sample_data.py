"""
Sample Data Insertion Script
==============================
Purpose: Insert sample data into all tables for testing and development

Usage:
    python database/insert_sample_data.py

What this script does:
1. Inserts realistic sample BGP data
2. Creates sample features
3. Generates sample ML results
4. Creates sample alerts
5. Useful for testing API and dashboard without real data
"""

import os
import psycopg2
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

load_dotenv()

class SampleDataInserter:
    """Insert sample data for testing"""
    
    def __init__(self):
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        self.conn = None
        self.cursor = None
        
        # Sample peer addresses (realistic BGP peers)
        self.sample_peers = [
            '203.0.113.1',    # Example peer 1
            '198.51.100.5',   # Example peer 2
            '192.0.2.10',     # Example peer 3
            '203.0.113.50',   # Example peer 4
            '198.51.100.100', # Example peer 5
        ]
        
        # Sample ASNs
        self.sample_asns = [64512, 64513, 64514, 64515, 64516]
        
        # Sample prefixes
        self.sample_prefixes = [
            '10.0.0.0/8',
            '172.16.0.0/12',
            '192.168.0.0/16',
            '203.0.113.0/24',
            '198.51.100.0/24',
        ]
    
    def connect(self):
        """Connect to database"""
        try:
            print("🔌 Connecting to database...")
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("✅ Connected!")
            return True
        except psycopg2.Error as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def insert_raw_bgp_data(self, num_records=100):
        """Insert sample raw BGP data"""
        print(f"\n📊 Inserting {num_records} raw BGP records...")
        
        base_time = datetime.now() - timedelta(hours=2)
        
        try:
            for i in range(num_records):
                timestamp = base_time + timedelta(seconds=i*30)  # Every 30 seconds
                peer_addr = random.choice(self.sample_peers)
                peer_asn = random.choice(self.sample_asns)
                prefix = random.choice(self.sample_prefixes)
                
                # Generate realistic feature values
                announcements = random.randint(0, 50)
                withdrawals = random.randint(0, 20)
                total_updates = announcements + withdrawals
                withdrawal_ratio = withdrawals / max(announcements, 1)
                flap_count = random.randint(0, 5)
                path_length = random.uniform(2.0, 8.0)
                unique_peers = random.randint(1, 10)
                message_rate = total_updates / 60.0
                session_resets = 1 if random.random() < 0.05 else 0  # 5% chance
                
                self.cursor.execute("""
                    INSERT INTO raw_bgp_data (
                        timestamp, peer_addr, peer_asn, prefix,
                        announcements, withdrawals, total_updates,
                        withdrawal_ratio, flap_count, path_length,
                        unique_peers, message_rate, session_resets
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    timestamp, peer_addr, peer_asn, prefix,
                    announcements, withdrawals, total_updates,
                    withdrawal_ratio, flap_count, path_length,
                    unique_peers, message_rate, session_resets
                ))
            
            self.conn.commit()
            print(f"✅ Inserted {num_records} raw BGP records")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Failed to insert raw BGP data: {e}")
            self.conn.rollback()
            return False
    
    def insert_features(self, num_records=50):
        """Insert sample aggregated features"""
        print(f"\n📊 Inserting {num_records} feature records...")
        
        base_time = datetime.now() - timedelta(hours=1)
        
        try:
            for i in range(num_records):
                timestamp = base_time + timedelta(minutes=i)  # Every minute
                peer_addr = random.choice(self.sample_peers)
                peer_asn = random.choice(self.sample_asns)
                
                # Aggregated features (higher values for anomalies)
                is_anomaly = random.random() < 0.2  # 20% anomalies
                
                if is_anomaly:
                    announcements = random.randint(500, 2000)
                    withdrawals = random.randint(200, 800)
                    flap_count = random.randint(50, 150)
                    path_length = random.uniform(15.0, 30.0)
                else:
                    announcements = random.randint(10, 100)
                    withdrawals = random.randint(0, 30)
                    flap_count = random.randint(0, 10)
                    path_length = random.uniform(2.0, 8.0)
                
                total_updates = announcements + withdrawals
                withdrawal_ratio = withdrawals / max(announcements, 1)
                unique_peers = random.randint(5, 20)
                message_rate = total_updates / 60.0
                session_resets = 1 if is_anomaly and random.random() < 0.3 else 0
                std_path_length = random.uniform(0.5, 3.0)
                max_updates = int(total_updates * 1.2)
                
                self.cursor.execute("""
                    INSERT INTO features (
                        timestamp, peer_addr, peer_asn, window_duration,
                        announcements, withdrawals, total_updates,
                        withdrawal_ratio, flap_count, path_length,
                        unique_peers, message_rate, session_resets,
                        std_path_length, max_updates
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    timestamp, peer_addr, peer_asn, 60,
                    announcements, withdrawals, total_updates,
                    withdrawal_ratio, flap_count, path_length,
                    unique_peers, message_rate, session_resets,
                    std_path_length, max_updates
                ))
            
            self.conn.commit()
            print(f"✅ Inserted {num_records} feature records")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Failed to insert features: {e}")
            self.conn.rollback()
            return False
    
    def insert_ml_results(self, num_records=50):
        """Insert sample ML detection results"""
        print(f"\n📊 Inserting {num_records} ML result records...")
        
        base_time = datetime.now() - timedelta(hours=1)
        
        try:
            for i in range(num_records):
                timestamp = base_time + timedelta(minutes=i)
                peer_addr = random.choice(self.sample_peers)
                
                # Simulate detection results
                is_anomaly = random.random() < 0.2
                
                if is_anomaly:
                    lstm_error = random.uniform(0.7, 1.5)
                    lstm_score = random.uniform(0.7, 1.0)
                    if_score = random.uniform(-0.8, -0.3)
                    heuristic_score = random.uniform(0.6, 1.0)
                    ensemble_score = random.uniform(0.7, 0.95)
                    ensemble_confidence = random.uniform(0.7, 0.9)
                    
                    # Random heuristic reasons
                    all_reasons = ['CRITICAL_CHURN', 'HIGH_CHURN', 'SEVERE_PATH_LEN', 
                                  'MASS_WITHDRAWAL', 'CRITICAL_FLAP']
                    heuristic_reasons = random.sample(all_reasons, random.randint(1, 3))
                else:
                    lstm_error = random.uniform(0.1, 0.4)
                    lstm_score = random.uniform(0.0, 0.3)
                    if_score = random.uniform(0.2, 0.8)
                    heuristic_score = random.uniform(0.0, 0.3)
                    ensemble_score = random.uniform(0.0, 0.4)
                    ensemble_confidence = random.uniform(0.5, 0.8)
                    heuristic_reasons = []
                
                self.cursor.execute("""
                    INSERT INTO ml_results (
                        timestamp, peer_addr, feature_id,
                        lstm_reconstruction_error, lstm_anomaly_score, lstm_is_anomaly,
                        if_anomaly_score, if_is_anomaly,
                        heuristic_score, heuristic_reasons, heuristic_is_anomaly,
                        ensemble_score, ensemble_confidence,
                        model_version, processing_time_ms
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    timestamp, peer_addr, i+1,
                    lstm_error, lstm_score, is_anomaly,
                    if_score, is_anomaly,
                    heuristic_score, heuristic_reasons, is_anomaly,
                    ensemble_score, ensemble_confidence,
                    'v1.0.0', random.uniform(50, 200)
                ))
            
            self.conn.commit()
            print(f"✅ Inserted {num_records} ML result records")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Failed to insert ML results: {e}")
            self.conn.rollback()
            return False
    
    def insert_route_monitor_events(self, num_records=20):
        """Insert sample RPKI validation events"""
        print(f"\n📊 Inserting {num_records} route monitor events...")
        
        base_time = datetime.now() - timedelta(hours=1)
        rpki_statuses = ['valid', 'invalid', 'unknown']
        event_types = ['rpki_invalid', 'hijack_suspected', 'leak_suspected']
        severities = ['critical', 'high', 'medium', 'low']
        
        try:
            for i in range(num_records):
                timestamp = base_time + timedelta(minutes=i*3)
                peer_addr = random.choice(self.sample_peers)
                peer_asn = random.choice(self.sample_asns)
                prefix = random.choice(self.sample_prefixes)
                origin_asn = random.choice(self.sample_asns)
                
                rpki_status = random.choice(rpki_statuses)
                event_type = random.choice(event_types)
                severity = random.choice(severities)
                
                description = f"{event_type.replace('_', ' ').title()} detected for {prefix}"
                as_path = ' '.join([str(random.choice(self.sample_asns)) for _ in range(3)])
                
                self.cursor.execute("""
                    INSERT INTO route_monitor_events (
                        timestamp, peer_addr, peer_asn, prefix, origin_asn,
                        rpki_status, event_type, severity, description, as_path
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    timestamp, peer_addr, peer_asn, prefix, origin_asn,
                    rpki_status, event_type, severity, description, as_path
                ))
            
            self.conn.commit()
            print(f"✅ Inserted {num_records} route monitor events")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Failed to insert route monitor events: {e}")
            self.conn.rollback()
            return False
    
    def insert_alerts(self, num_records=15):
        """Insert sample alerts"""
        print(f"\n📊 Inserting {num_records} alert records...")
        
        base_time = datetime.now() - timedelta(minutes=30)
        alert_types = ['ml_anomaly', 'rpki_invalid', 'hybrid']
        severities = ['critical', 'high', 'medium', 'low']
        statuses = ['open', 'acknowledged', 'resolved']
        anomaly_types = ['churn', 'path_length', 'withdrawal', 'flapping', 'rpki_invalid']
        
        try:
            for i in range(num_records):
                timestamp = base_time + timedelta(minutes=i*2)
                peer_addr = random.choice(self.sample_peers)
                peer_asn = random.choice(self.sample_asns)
                
                alert_type = random.choice(alert_types)
                severity = random.choice(severities)
                confidence = random.uniform(0.6, 0.95)
                status = random.choices(statuses, weights=[0.5, 0.3, 0.2])[0]
                
                selected_anomaly_types = random.sample(anomaly_types, random.randint(1, 3))
                
                title = f"{severity.upper()}: {alert_type.replace('_', ' ').title()} on {peer_addr}"
                description = f"Detected {', '.join(selected_anomaly_types)} anomaly patterns"
                
                ensemble_score = random.uniform(0.6, 0.95) if severity in ['critical', 'high'] else random.uniform(0.4, 0.7)
                final_score = ensemble_score * confidence
                
                self.cursor.execute("""
                    INSERT INTO alerts (
                        timestamp, alert_type, severity, confidence,
                        peer_addr, peer_asn, affected_prefixes,
                        title, description, anomaly_types,
                        ml_result_id, ensemble_score, final_score,
                        status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s::cidr[], %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    timestamp, alert_type, severity, confidence,
                    peer_addr, peer_asn, [random.choice(self.sample_prefixes)],
                    title, description, selected_anomaly_types,
                    i+1, ensemble_score, final_score, status
                ))
            
            self.conn.commit()
            print(f"✅ Inserted {num_records} alert records")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Failed to insert alerts: {e}")
            self.conn.rollback()
            return False
    
    def insert_system_metrics(self, num_records=30):
        """Insert sample system metrics"""
        print(f"\n📊 Inserting {num_records} system metric records...")
        
        base_time = datetime.now() - timedelta(minutes=30)
        components = ['ris_collector', 'feature_aggregator', 'ml_detector', 'correlation_engine', 'api']
        metric_names = ['ingestion_rate', 'detection_latency', 'memory_usage', 'cpu_usage']
        
        try:
            for i in range(num_records):
                timestamp = base_time + timedelta(minutes=i)
                component = random.choice(components)
                metric_name = random.choice(metric_names)
                
                if metric_name == 'ingestion_rate':
                    value = random.uniform(10, 100)
                    unit = 'records/sec'
                elif metric_name == 'detection_latency':
                    value = random.uniform(50, 500)
                    unit = 'milliseconds'
                elif metric_name == 'memory_usage':
                    value = random.uniform(30, 80)
                    unit = 'percent'
                else:  # cpu_usage
                    value = random.uniform(10, 70)
                    unit = 'percent'
                
                self.cursor.execute("""
                    INSERT INTO system_metrics (
                        timestamp, metric_name, metric_value, unit, component
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (timestamp, metric_name, value, unit, component))
            
            self.conn.commit()
            print(f"✅ Inserted {num_records} system metric records")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Failed to insert system metrics: {e}")
            self.conn.rollback()
            return False
    
    def show_summary(self):
        """Show summary of inserted data"""
        print("\n" + "=" * 60)
        print("📊 DATA SUMMARY")
        print("=" * 60)
        
        tables = [
            'raw_bgp_data',
            'features',
            'ml_results',
            'route_monitor_events',
            'alerts',
            'system_metrics'
        ]
        
        try:
            for table in tables:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = self.cursor.fetchone()[0]
                print(f"   {table}: {count} records")
            
            print("=" * 60)
            
        except psycopg2.Error as e:
            print(f"⚠️  Could not generate summary: {e}")
    
    def close(self):
        """Close connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("\n🔌 Connection closed")
    
    def run(self):
        """Main execution"""
        print("=" * 60)
        print("INSERTING SAMPLE DATA")
        print("=" * 60)
        
        if not self.connect():
            return False
        
        # Insert data in order (respecting dependencies)
        success = (
            self.insert_raw_bgp_data(100) and
            self.insert_features(50) and
            self.insert_ml_results(50) and
            self.insert_route_monitor_events(20) and
            self.insert_alerts(15) and
            self.insert_system_metrics(30)
        )
        
        if success:
            self.show_summary()
            print("\n✅ Sample data insertion complete!")
            print("\n💡 You can now:")
            print("   - Query data: SELECT * FROM alerts LIMIT 10;")
            print("   - Start API: python api/main.py")
            print("   - Open dashboard in browser")
        else:
            print("\n❌ Sample data insertion failed")
        
        return success


def main():
    inserter = SampleDataInserter()
    try:
        inserter.run()
    finally:
        inserter.close()


if __name__ == "__main__":
    main()
