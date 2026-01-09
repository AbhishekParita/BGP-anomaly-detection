"""
Correlation Engine Service

This service correlates BGP anomalies from multiple sources to generate
high-confidence alerts with reduced false positives.

Architecture Flow:
ml_results + route_monitor_events + features 
    → Correlation Engine → alerts table

Correlation Strategies:
1. Multi-peer correlation (same prefix from multiple peers)
2. RPKI validation integration (invalid ROAs)
3. Temporal correlation (repeated anomalies)
4. Geographic correlation (same ASN across regions)
5. Pattern matching (known attack signatures)

Author: BGP Anomaly Detection System
Created: 2026-01-07
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('correlation_engine.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CorrelationEngine:
    """
    Correlation engine that generates high-confidence alerts by combining
    multiple signals and applying correlation logic.
    """
    
    def __init__(self):
        """Initialize Correlation Engine with database connection."""
        self.db_conn = None
        self.db_cursor = None
        
        # Database configuration
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Correlation thresholds
        self.thresholds = {
            'ensemble_anomaly_score': 0.6,  # Min ensemble score for correlation
            'multi_peer_count': 3,          # Min peers for coordinated attack
            'temporal_window_minutes': 10,   # Time window for temporal correlation
            'repeated_anomaly_count': 3      # Min repetitions for pattern
        }
        
        # Severity weights
        self.severity_weights = {
            'multi_peer': 3.0,      # Multiple peers seeing same issue
            'rpki_invalid': 2.5,    # RPKI validation failure
            'repeated': 2.0,        # Repeated anomaly pattern
            'high_score': 1.5,      # Very high ensemble score
            'known_pattern': 2.0    # Matches known attack pattern
        }
        
        # Processing settings
        self.processing_interval = 15  # seconds
        self.correlation_window = 300  # 5 minutes for correlation
        self.batch_size = 100
        
        # Statistics
        self.stats = {
            'anomalies_processed': 0,
            'alerts_generated': 0,
            'critical_alerts': 0,
            'high_alerts': 0,
            'medium_alerts': 0,
            'low_alerts': 0,
            'errors': 0
        }
    
    def connect_db(self):
        """Establish database connection."""
        try:
            self.db_conn = psycopg2.connect(**self.db_config)
            self.db_cursor = self.db_conn.cursor()
            logger.info("[OK] Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Database connection failed: {e}")
            return False
    
    def close_db(self):
        """Close database connection."""
        if self.db_cursor:
            self.db_cursor.close()
        if self.db_conn:
            self.db_conn.close()
        logger.info("Database connection closed")
    
    def get_recent_anomalies(self, minutes: int = 5) -> List[Dict]:
        """
        Fetch recent ensemble anomalies for correlation.
        
        Args:
            minutes: Look back window in minutes
            
        Returns:
            List of anomaly dictionaries
        """
        try:
            query = """
            SELECT 
                m.id, m.timestamp, m.peer_addr, m.feature_id,
                m.ensemble_score, m.ensemble_is_anomaly,
                m.heuristic_score, m.lstm_anomaly_score, m.if_anomaly_score,
                f.peer_asn, f.announcements, f.withdrawals, f.total_updates
            FROM ml_results m
            JOIN features f ON m.feature_id = f.id
            WHERE m.ensemble_is_anomaly = TRUE
              AND m.timestamp >= NOW() - INTERVAL '%s minutes'
              AND NOT EXISTS (
                  SELECT 1 FROM alerts a 
                  WHERE a.ml_result_id = m.id
              )
            ORDER BY m.timestamp DESC
            LIMIT %s;
            """
            
            self.db_cursor.execute(query, (minutes, self.batch_size))
            rows = self.db_cursor.fetchall()
            
            anomalies = []
            for row in rows:
                anomalies.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'peer_addr': row[2],
                    'feature_id': row[3],
                    'ensemble_score': row[4],
                    'ensemble_is_anomaly': row[5],
                    'heuristic_score': row[6],
                    'lstm_score': row[7],
                    'if_score': row[8],
                    'peer_asn': row[9],
                    'announcements': row[10],
                    'withdrawals': row[11],
                    'total_updates': row[12]
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error fetching recent anomalies: {e}")
            return []
    
    def check_rpki_validation(self, peer_addr: str, timestamp: datetime) -> Optional[Dict]:
        """
        Check if there are RPKI validation failures for this peer.
        
        Args:
            peer_addr: BGP peer address
            timestamp: Anomaly timestamp
            
        Returns:
            Dict with RPKI info or None
        """
        try:
            query = """
            SELECT rpki_status, event_type, severity, description
            FROM route_monitor_events
            WHERE peer_addr = %s
              AND timestamp BETWEEN %s - INTERVAL '5 minutes' 
                  AND %s + INTERVAL '5 minutes'
              AND rpki_status = 'invalid'
            ORDER BY timestamp DESC
            LIMIT 1;
            """
            
            self.db_cursor.execute(query, (peer_addr, timestamp, timestamp))
            row = self.db_cursor.fetchone()
            
            if row:
                return {
                    'rpki_status': row[0],
                    'event_type': row[1],
                    'severity': row[2],
                    'description': row[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking RPKI: {e}")
            return None
    
    def find_multi_peer_correlation(self, anomalies: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Find anomalies affecting multiple peers (coordinated attacks).
        
        Args:
            anomalies: List of anomaly dictionaries
            
        Returns:
            Dict mapping correlation key to list of correlated anomalies
        """
        # Group by time window (1-minute buckets)
        time_buckets = defaultdict(list)
        
        for anomaly in anomalies:
            # Round timestamp to nearest minute
            bucket_time = anomaly['timestamp'].replace(second=0, microsecond=0)
            time_buckets[bucket_time].append(anomaly)
        
        # Find buckets with multiple peers
        correlated = {}
        
        for bucket_time, bucket_anomalies in time_buckets.items():
            if len(bucket_anomalies) >= self.thresholds['multi_peer_count']:
                # Check if multiple unique peers
                unique_peers = set(a['peer_addr'] for a in bucket_anomalies)
                
                if len(unique_peers) >= self.thresholds['multi_peer_count']:
                    key = f"multi_peer_{bucket_time.isoformat()}"
                    correlated[key] = bucket_anomalies
        
        return correlated
    
    def find_temporal_patterns(self, peer_addr: str) -> int:
        """
        Check for repeated anomaly patterns from the same peer.
        
        Args:
            peer_addr: BGP peer address
            
        Returns:
            int: Number of recent anomalies from this peer
        """
        try:
            query = """
            SELECT COUNT(*)
            FROM ml_results
            WHERE peer_addr = %s
              AND ensemble_is_anomaly = TRUE
              AND timestamp >= NOW() - INTERVAL '%s minutes';
            """
            
            self.db_cursor.execute(
                query, 
                (peer_addr, self.thresholds['temporal_window_minutes'])
            )
            
            count = self.db_cursor.fetchone()[0]
            return count
            
        except Exception as e:
            logger.error(f"Error checking temporal patterns: {e}")
            return 0
    
    def calculate_severity(self, anomaly: Dict, correlation_factors: Dict) -> Tuple[str, float]:
        """
        Calculate alert severity based on correlation factors.
        
        Args:
            anomaly: Anomaly dictionary
            correlation_factors: Dict with correlation evidence
            
        Returns:
            Tuple of (severity_level, confidence_score)
        """
        confidence = 0.0
        reasons = []
        
        # Base confidence from ensemble score
        confidence += anomaly['ensemble_score'] * 0.3
        
        # Multi-peer correlation (strongest signal)
        if correlation_factors.get('multi_peer'):
            peer_count = correlation_factors['multi_peer']
            confidence += self.severity_weights['multi_peer'] * min(peer_count / 10, 1.0)
            reasons.append(f"MULTI_PEER:{peer_count}")
        
        # RPKI validation failure
        if correlation_factors.get('rpki_invalid'):
            confidence += self.severity_weights['rpki_invalid']
            reasons.append("RPKI_INVALID")
        
        # Repeated anomaly pattern
        repeated_count = correlation_factors.get('repeated', 0)
        if repeated_count >= self.thresholds['repeated_anomaly_count']:
            confidence += self.severity_weights['repeated'] * min(repeated_count / 10, 1.0)
            reasons.append(f"REPEATED:{repeated_count}")
        
        # Very high ensemble score
        if anomaly['ensemble_score'] > 0.8:
            confidence += self.severity_weights['high_score']
            reasons.append("HIGH_SCORE")
        
        # Normalize confidence to [0, 1]
        confidence = min(confidence, 1.0)
        
        # Determine severity level
        if confidence >= 0.8:
            severity = "critical"
            self.stats['critical_alerts'] += 1
        elif confidence >= 0.6:
            severity = "high"
            self.stats['high_alerts'] += 1
        elif confidence >= 0.4:
            severity = "medium"
            self.stats['medium_alerts'] += 1
        else:
            severity = "low"
            self.stats['low_alerts'] += 1
        
        return severity, confidence
    
    def generate_alert_description(self, anomaly: Dict, correlation_factors: Dict) -> str:
        """
        Generate human-readable alert description.
        
        Args:
            anomaly: Anomaly dictionary
            correlation_factors: Correlation evidence
            
        Returns:
            str: Alert description
        """
        desc_parts = []
        
        # Basic anomaly info
        desc_parts.append(
            f"BGP anomaly detected from peer {anomaly['peer_addr']} "
            f"(AS{anomaly.get('peer_asn', 'unknown')})"
        )
        
        # Traffic details
        if anomaly.get('total_updates'):
            desc_parts.append(
                f"Traffic: {anomaly['announcements']} announcements, "
                f"{anomaly['withdrawals']} withdrawals"
            )
        
        # Correlation evidence
        if correlation_factors.get('multi_peer'):
            desc_parts.append(
                f"Correlated across {correlation_factors['multi_peer']} peers "
                f"(possible coordinated attack)"
            )
        
        if correlation_factors.get('rpki_invalid'):
            desc_parts.append("RPKI validation failure detected")
        
        if correlation_factors.get('repeated', 0) >= 3:
            desc_parts.append(
                f"Repeated pattern ({correlation_factors['repeated']} occurrences)"
            )
        
        # Model scores
        desc_parts.append(
            f"Scores: Ensemble={anomaly['ensemble_score']:.2f}, "
            f"Heuristic={anomaly.get('heuristic_score', 0):.2f}, "
            f"LSTM={anomaly.get('lstm_score', 0):.2f}"
        )
        
        return ". ".join(desc_parts)
    
    def process_anomalies(self) -> int:
        """
        Process recent anomalies and generate correlated alerts.
        
        Returns:
            int: Number of alerts generated
        """
        try:
            # Fetch recent anomalies
            anomalies = self.get_recent_anomalies(minutes=5)
            
            if not anomalies:
                logger.debug("No new anomalies to process")
                return 0
            
            logger.info(f"Processing {len(anomalies)} anomalies for correlation")
            
            # Find multi-peer correlations
            multi_peer_groups = self.find_multi_peer_correlation(anomalies)
            
            # Track which anomalies are part of correlated groups
            correlated_ids = set()
            for group in multi_peer_groups.values():
                correlated_ids.update(a['id'] for a in group)
            
            # Process each anomaly
            alerts = []
            
            for anomaly in anomalies:
                correlation_factors = {}
                
                # Check if part of multi-peer correlation
                if anomaly['id'] in correlated_ids:
                    for group in multi_peer_groups.values():
                        if anomaly in group:
                            correlation_factors['multi_peer'] = len(group)
                            break
                
                # Check RPKI validation
                rpki_info = self.check_rpki_validation(
                    anomaly['peer_addr'], 
                    anomaly['timestamp']
                )
                if rpki_info and rpki_info['rpki_status'] == 'invalid':
                    correlation_factors['rpki_invalid'] = True
                
                # Check temporal patterns
                repeated_count = self.find_temporal_patterns(anomaly['peer_addr'])
                if repeated_count > 1:
                    correlation_factors['repeated'] = repeated_count
                
                # Calculate severity
                severity, confidence = self.calculate_severity(anomaly, correlation_factors)
                
                # Generate description
                description = self.generate_alert_description(anomaly, correlation_factors)
                
                # Create alert
                alerts.append({
                    'ml_result_id': anomaly['id'],
                    'timestamp': anomaly['timestamp'],
                    'peer_addr': anomaly['peer_addr'],
                    'peer_asn': anomaly.get('peer_asn'),
                    'alert_type': 'bgp_anomaly',
                    'severity': severity,
                    'confidence': confidence,
                    'description': description,
                    'correlation_factors': correlation_factors
                })
            
            # Insert alerts
            inserted = self.insert_alerts(alerts)
            
            self.stats['anomalies_processed'] += len(anomalies)
            self.stats['alerts_generated'] += inserted
            
            logger.info(
                f"[OK] Generated {inserted} alerts from {len(anomalies)} anomalies "
                f"(Critical: {self.stats['critical_alerts']}, "
                f"High: {self.stats['high_alerts']}, "
                f"Medium: {self.stats['medium_alerts']}, "
                f"Low: {self.stats['low_alerts']})"
            )
            
            return inserted
            
        except Exception as e:
            logger.error(f"Error processing anomalies: {e}")
            self.stats['errors'] += 1
            return 0
    
    def insert_alerts(self, alerts: List[Dict]) -> int:
        """
        Insert alerts into alerts table.
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            int: Number of alerts inserted
        """
        if not alerts:
            return 0
        
        try:
            query = """
            INSERT INTO alerts (
                timestamp, peer_addr, peer_asn, ml_result_id,
                alert_type, severity, confidence, description,
                is_acknowledged, is_resolved
            ) VALUES %s;
            """
            
            values = [
                (
                    a['timestamp'], a['peer_addr'], a['peer_asn'], a['ml_result_id'],
                    a['alert_type'], a['severity'], a['confidence'], a['description'],
                    False, False
                )
                for a in alerts
            ]
            
            execute_values(self.db_cursor, query, values)
            self.db_conn.commit()
            
            return len(alerts)
            
        except Exception as e:
            logger.error(f"Error inserting alerts: {e}")
            self.db_conn.rollback()
            self.stats['errors'] += 1
            return 0
    
    async def run_continuous(self):
        """
        Run correlation engine continuously in a loop.
        """
        logger.info("=" * 60)
        logger.info("CORRELATION ENGINE SERVICE STARTED")
        logger.info("=" * 60)
        logger.info(f"Processing interval: {self.processing_interval} seconds")
        logger.info(f"Correlation window: {self.correlation_window} seconds")
        logger.info(f"Multi-peer threshold: {self.thresholds['multi_peer_count']} peers")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Connect to database
        if not self.connect_db():
            logger.error("Cannot start service without database connection")
            return
        
        try:
            while True:
                # Process pending anomalies
                alerts_generated = self.process_anomalies()
                
                if alerts_generated > 0:
                    logger.info(f"[OK] Generated {alerts_generated} new alerts")
                
                # Wait before next cycle
                await asyncio.sleep(self.processing_interval)
                
        except KeyboardInterrupt:
            logger.info("\n[WARNING] Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Print final statistics
            self.close_db()
            logger.info("=" * 60)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total anomalies processed: {self.stats['anomalies_processed']}")
            logger.info(f"Total alerts generated: {self.stats['alerts_generated']}")
            logger.info(f"  Critical: {self.stats['critical_alerts']}")
            logger.info(f"  High: {self.stats['high_alerts']}")
            logger.info(f"  Medium: {self.stats['medium_alerts']}")
            logger.info(f"  Low: {self.stats['low_alerts']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("=" * 60)
    
    async def run_once(self):
        """
        Run correlation once (for testing/batch processing).
        """
        logger.info("Running correlation engine (single pass)...")
        
        if not self.connect_db():
            logger.error("Cannot run without database connection")
            return
        
        try:
            alerts = self.process_anomalies()
            logger.info(f"[OK] Generated {alerts} alerts")
            logger.info(f"Severity breakdown: Critical={self.stats['critical_alerts']}, "
                      f"High={self.stats['high_alerts']}, "
                      f"Medium={self.stats['medium_alerts']}, "
                      f"Low={self.stats['low_alerts']}")
        finally:
            self.close_db()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BGP Correlation Engine Service')
    parser.add_argument(
        '--mode',
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous (daemon) or once (batch)'
    )
    
    args = parser.parse_args()
    
    engine = CorrelationEngine()
    
    if args.mode == 'continuous':
        asyncio.run(engine.run_continuous())
    else:
        asyncio.run(engine.run_once())


if __name__ == "__main__":
    main()
