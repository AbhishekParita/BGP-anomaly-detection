"""
RIS Live Collector Service
===========================
Purpose: Connect to RIPE RIS Live WebSocket, parse BGP messages, extract features,
         and insert into PostgreSQL raw_bgp_data table

Data Flow:
    RIS Live WebSocket → Parse BGP Message → Extract 9 Features → raw_bgp_data table

What this service does:
1. Connects to wss://ris-live.ripe.net/v1/ws
2. Subscribes to BGP UPDATE messages from all collectors
3. Parses each message to extract:
   - Peer information
   - Announcements and withdrawals
   - AS path details
   - Flapping information
4. Calculates 9 core features
5. Inserts into raw_bgp_data table in real-time

Usage:
    python services/ris_live_collector.py
"""

import asyncio
import websockets
import json
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import os
import sys
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RISLiveCollector:
    """
    Collects BGP data from RIS Live and stores in PostgreSQL
    
    Architecture:
    - Async WebSocket connection to RIS Live
    - Message parsing and feature extraction
    - Batched database inserts for performance
    """
    
    def __init__(self):
        """Initialize collector with database connection and state tracking"""
        
        # Database configuration
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'anand'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # RIS Live WebSocket URL (optimized subscription)
        self.ris_live_url = "wss://ris-live.ripe.net/v1/ws/?client=bgp-anomaly-detector"
        
        # Database connection
        self.conn = None
        self.cursor = None
        
        # Message statistics
        self.stats = {
            'total_messages': 0,
            'announcements': 0,
            'withdrawals': 0,
            'errors': 0,
            'inserts': 0
        }
        
        # Peer state tracking for feature calculation
        # Format: {peer_addr: {'last_update': timestamp, 'flap_count': int, ...}}
        self.peer_state = defaultdict(lambda: {
            'last_update': None,
            'flap_count': 0,
            'session_resets': 0,
            'announcement_count': 0,
            'withdrawal_count': 0
        })
        
        # Batch insert buffer (increased for faster collection)
        self.insert_buffer = []
        self.buffer_size = 50  # Insert every 50 messages (faster processing)
        
        logger.info("RIS Live Collector initialized")
    
    def connect_db(self):
        """Establish database connection"""
        try:
            logger.info(f"Connecting to PostgreSQL at {self.db_config['host']}:{self.db_config['port']}")
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("✅ Database connected successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def close_db(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    async def subscribe_to_ris_live(self, websocket):
        """
        Subscribe to RIS Live BGP updates
        
        Subscription message:
        - type: UPDATE (only BGP updates, not status messages)
        - host: None (all collectors)
        - socketOptions: no raw data, send acknowledgments
        """
        subscribe_message = {
            "type": "ris_subscribe",
            "data": {
                "host": None,  # Subscribe to all RIS collectors
                "type": "UPDATE",  # Only BGP UPDATE messages
                "socketOptions": {
                    "includeRaw": False,  # Don't include raw BGP data
                    "acknowledge": True   # Send acknowledgments
                }
            }
        }
        
        await websocket.send(json.dumps(subscribe_message))
        logger.info("📡 Subscribed to RIS Live BGP updates")
    
    def extract_features_from_message(self, message):
        """
        Extract 9 core features from RIS Live BGP message
        
        Input: RIS Live message JSON
        Output: Dictionary with 9 features
        
        Features:
        1. announcements - Number of route announcements
        2. withdrawals - Number of route withdrawals
        3. total_updates - announcements + withdrawals
        4. withdrawal_ratio - withdrawals / max(announcements, 1)
        5. flap_count - Route flapping incidents
        6. path_length - Average AS path length
        7. unique_peers - Number of unique peers (1 per message)
        8. message_rate - Messages per second (calculated over time)
        9. session_resets - BGP session reset count
        """
        
        try:
            # Extract timestamp
            timestamp = datetime.fromtimestamp(message.get('timestamp', 0))
            
            # Extract peer information
            peer_addr = message.get('peer', 'unknown')
            peer_asn = message.get('peer_asn')
            
            # Extract announcements and withdrawals
            announcements = message.get('announcements', [])
            withdrawals = message.get('withdrawals', [])
            
            num_announcements = len(announcements)
            num_withdrawals = len(withdrawals)
            total_updates = num_announcements + num_withdrawals
            
            # Calculate withdrawal ratio
            withdrawal_ratio = num_withdrawals / max(num_announcements, 1)
            
            # Extract AS path and calculate average path length
            path = message.get('path', [])
            path_length = len(path) if path else 0
            
            # Get prefix (first announcement or withdrawal)
            prefix = None
            if announcements:
                # Handle both string prefixes and dict formats
                first_ann = announcements[0]
                if isinstance(first_ann, str):
                    prefix = first_ann
                elif isinstance(first_ann, dict):
                    prefixes = first_ann.get('prefixes', [])
                    prefix = prefixes[0] if prefixes else None
            elif withdrawals:
                # Handle both string prefixes and dict formats
                first_with = withdrawals[0]
                if isinstance(first_with, str):
                    prefix = first_with
                elif isinstance(first_with, dict):
                    prefixes = first_with.get('prefixes', [])
                    prefix = prefixes[0] if prefixes else None
            
            # Update peer state for flapping and session tracking
            peer_info = self.peer_state[peer_addr]
            
            # Check for flapping (frequent updates to same peer)
            if peer_info['last_update']:
                time_diff = (timestamp - peer_info['last_update']).total_seconds()
                if time_diff < 60:  # Updates within 1 minute indicate flapping
                    peer_info['flap_count'] += 1
            
            peer_info['last_update'] = timestamp
            peer_info['announcement_count'] += num_announcements
            peer_info['withdrawal_count'] += num_withdrawals
            
            # Session resets detection (simplified - would need state machine in production)
            session_resets = peer_info['session_resets']
            
            # Message rate (simplified - messages per minute)
            message_rate = total_updates / 60.0
            
            # Return extracted features
            return {
                'timestamp': timestamp,
                'peer_addr': peer_addr,
                'peer_asn': peer_asn,
                'prefix': prefix,
                'announcements': num_announcements,
                'withdrawals': num_withdrawals,
                'total_updates': total_updates,
                'withdrawal_ratio': withdrawal_ratio,
                'flap_count': peer_info['flap_count'],
                'path_length': float(path_length),
                'unique_peers': 1,  # Each message is from one peer
                'message_rate': message_rate,
                'session_resets': session_resets,
                'raw_message': json.dumps(message)  # Store full message as JSON
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def insert_to_database(self, features):
        """
        Insert extracted features into raw_bgp_data table
        
        Uses batched inserts for better performance
        """
        if not features:
            return
        
        # Add to buffer
        self.insert_buffer.append(features)
        
        # Insert when buffer is full
        if len(self.insert_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush insert buffer to database"""
        if not self.insert_buffer:
            return
        
        try:
            # Prepare batch insert
            insert_query = """
                INSERT INTO raw_bgp_data (
                    timestamp, peer_addr, peer_asn, prefix,
                    announcements, withdrawals, total_updates,
                    withdrawal_ratio, flap_count, path_length,
                    unique_peers, message_rate, session_resets,
                    raw_message
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            # Execute batch insert
            for features in self.insert_buffer:
                self.cursor.execute(insert_query, (
                    features['timestamp'],
                    features['peer_addr'],
                    features['peer_asn'],
                    features['prefix'],
                    features['announcements'],
                    features['withdrawals'],
                    features['total_updates'],
                    features['withdrawal_ratio'],
                    features['flap_count'],
                    features['path_length'],
                    features['unique_peers'],
                    features['message_rate'],
                    features['session_resets'],
                    features['raw_message']
                ))
            
            self.conn.commit()
            self.stats['inserts'] += len(self.insert_buffer)
            
            logger.info(f"✅ Inserted {len(self.insert_buffer)} records | Total: {self.stats['inserts']}")
            
            # Clear buffer
            self.insert_buffer.clear()
            
        except psycopg2.Error as e:
            logger.error(f"❌ Database insert failed: {e}")
            self.conn.rollback()
            self.stats['errors'] += 1
    
    async def process_message(self, message_str):
        """
        Process incoming RIS Live message
        
        Flow:
        1. Parse JSON
        2. Filter for ris_message type
        3. Extract features
        4. Insert to database
        """
        try:
            message = json.loads(message_str)
            
            # Only process ris_message type (actual BGP updates)
            if message.get('type') != 'ris_message':
                return
            
            # Extract message data
            data = message.get('data', {})
            
            # Update statistics
            self.stats['total_messages'] += 1
            
            if data.get('announcements'):
                self.stats['announcements'] += len(data['announcements'])
            if data.get('withdrawals'):
                self.stats['withdrawals'] += len(data['withdrawals'])
            
            # Extract features
            features = self.extract_features_from_message(data)
            
            if features:
                # Insert to database (buffered)
                self.insert_to_database(features)
            
            # Log progress every 100 messages
            if self.stats['total_messages'] % 100 == 0:
                logger.info(
                    f"📊 Stats: {self.stats['total_messages']} messages | "
                    f"{self.stats['announcements']} announcements | "
                    f"{self.stats['withdrawals']} withdrawals | "
                    f"{self.stats['inserts']} inserted"
                )
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.stats['errors'] += 1
    
    async def listen_to_ris_live(self):
        """
        Main event loop: Listen to RIS Live WebSocket and process messages
        
        Handles:
        - Connection establishment
        - Subscription
        - Message processing
        - Reconnection on errors
        """
        while True:
            try:
                logger.info(f"🌐 Connecting to RIS Live: {self.ris_live_url}")
                
                async with websockets.connect(self.ris_live_url) as websocket:
                    logger.info("✅ Connected to RIS Live!")
                    
                    # Subscribe to updates
                    await self.subscribe_to_ris_live(websocket)
                    
                    # Listen for messages
                    async for message in websocket:
                        await self.process_message(message)
            
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"❌ WebSocket error: {e}")
                logger.info("🔄 Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                logger.info("🔄 Reconnecting in 10 seconds...")
                await asyncio.sleep(10)
    
    async def run(self):
        """Start the collector service"""
        logger.info("=" * 60)
        logger.info("RIS LIVE COLLECTOR - STARTING")
        logger.info("=" * 60)
        
        # Connect to database
        if not self.connect_db():
            logger.error("Cannot start without database connection")
            return
        
        try:
            # Start listening
            await self.listen_to_ris_live()
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Shutdown requested by user")
        
        finally:
            # Flush any remaining buffer
            self.flush_buffer()
            
            # Close database
            self.close_db()
            
            # Print final statistics
            logger.info("=" * 60)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total messages processed: {self.stats['total_messages']}")
            logger.info(f"Announcements: {self.stats['announcements']}")
            logger.info(f"Withdrawals: {self.stats['withdrawals']}")
            logger.info(f"Records inserted: {self.stats['inserts']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("=" * 60)


def main():
    """Entry point"""
    collector = RISLiveCollector()
    
    try:
        # Run async event loop
        asyncio.run(collector.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
