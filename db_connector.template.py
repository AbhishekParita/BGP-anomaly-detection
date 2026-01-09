"""
Database Connector Template
Copy this file to db_connector.py and configure with your credentials
"""

import psycopg2
from psycopg2 import pool
import logging

logger = logging.getLogger(__name__)

# Database Configuration - REPLACE WITH YOUR VALUES
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'your_database_name',
    'user': 'your_username',
    'password': 'your_password'  # ⚠️ CHANGE THIS - DO NOT COMMIT YOUR REAL PASSWORD
}

# Connection Pool
connection_pool = None

def get_db_pool(minconn=1, maxconn=10):
    """Initialize database connection pool"""
    global connection_pool
    
    if connection_pool is None:
        try:
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn,
                maxconn,
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    
    return connection_pool

def get_db_connection():
    """Get a connection from the pool"""
    try:
        pool = get_db_pool()
        return pool.getconn()
    except Exception as e:
        logger.error(f"Error getting database connection: {e}")
        raise

def release_db_connection(conn):
    """Return connection to the pool"""
    try:
        pool = get_db_pool()
        pool.putconn(conn)
    except Exception as e:
        logger.error(f"Error releasing connection: {e}")

def close_db_pool():
    """Close all connections in the pool"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        logger.info("Database connection pool closed")

# Context manager for automatic connection handling
class DBConnection:
    def __enter__(self):
        self.conn = get_db_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release_db_connection(self.conn)
