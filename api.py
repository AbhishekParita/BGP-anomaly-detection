"""
BGP Anomaly Detection System - REST API

FastAPI-based REST API for querying anomaly detection results, alerts, and statistics.

Endpoints:
- GET /api/alerts - Query alerts with filters
- GET /api/alerts/{alert_id} - Get specific alert
- GET /api/statistics - Get system statistics
- GET /api/detections - Query ML detection results
- GET /api/peers - List monitored peers
- GET /api/health - Health check endpoint

Author: BGP Anomaly Detection System
Created: 2026-01-07
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from pathlib import Path as FilePath

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("BGP Anomaly Detection API started")
    logger.info("Documentation available at: http://localhost:8000/docs")
    
    # Create drift monitoring tables if they don't exist
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_flags (
                id SERIAL PRIMARY KEY,
                model VARCHAR(50) NOT NULL,
                drift_detected BOOLEAN NOT NULL,
                drift_score FLOAT,
                threshold FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details JSONB
            );
            
            CREATE TABLE IF NOT EXISTS drift_reports (
                id SERIAL PRIMARY KEY,
                model VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                baseline_start TIMESTAMP,
                baseline_end TIMESTAMP,
                current_start TIMESTAMP,
                current_end TIMESTAMP,
                metrics JSONB,
                status VARCHAR(20),
                action_taken VARCHAR(100)
            );
            
            CREATE INDEX IF NOT EXISTS idx_drift_flags_timestamp ON drift_flags(timestamp);
            CREATE INDEX IF NOT EXISTS idx_drift_flags_model ON drift_flags(model);
            CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp ON drift_reports(timestamp);
            CREATE INDEX IF NOT EXISTS idx_drift_reports_model ON drift_reports(model);
        """)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Drift monitoring tables verified/created")
    except Exception as e:
        logger.warning(f"Could not create drift tables: {e}")
    
    yield
    
    # Shutdown
    logger.info("BGP Anomaly Detection API stopped")

# Initialize FastAPI app
app = FastAPI(
    title="BGP Anomaly Detection API",
    description="REST API for querying BGP anomaly detection results and alerts",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (dashboard)
dashboard_path = FilePath(__file__).parent / "dashboard"
if dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_path)), name="dashboard")

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'bgp_monitor'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SeverityEnum(str, Enum):
    """Alert severity levels."""
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class AlertTypeEnum(str, Enum):
    """Alert types."""
    ml_anomaly = "ml_anomaly"
    rpki_invalid = "rpki_invalid"
    hijack_suspected = "hijack_suspected"
    route_leak = "route_leak"
    hybrid = "hybrid"


class Alert(BaseModel):
    """Alert response model."""
    id: int
    alert_type: str
    severity: str
    peer_addr: str
    title: str
    description: Optional[str]
    confidence: float
    final_score: float
    anomaly_types: Optional[List[str]] = []
    status: Optional[str] = "open"
    created_at: datetime


class Detection(BaseModel):
    """ML detection result model."""
    id: int
    timestamp: datetime
    peer_addr: str
    heuristic_score: Optional[float]
    heuristic_is_anomaly: Optional[bool]
    lstm_anomaly_score: Optional[float]
    lstm_is_anomaly: Optional[bool]
    if_anomaly_score: Optional[float]
    if_is_anomaly: Optional[bool]
    ensemble_score: float
    ensemble_is_anomaly: bool


class Statistics(BaseModel):
    """System statistics model."""
    total_alerts: int
    alerts_by_severity: Dict[str, int]
    total_detections: int
    anomaly_rate: float
    active_peers: int
    time_range: Dict[str, str]


class PeerInfo(BaseModel):
    """Peer information model."""
    peer_addr: str
    total_detections: int
    anomaly_count: int
    anomaly_rate: float
    last_seen: datetime


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    database: str
    timestamp: datetime


# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirect to dashboard."""
    return FileResponse("dashboard/index.html")


@app.get("/api/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Check API and database health."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        cursor.close()
        conn.close()
        
        return HealthStatus(
            status="healthy",
            database="connected",
            timestamp=datetime.now()
        )
    except Exception as e:
        return HealthStatus(
            status="unhealthy",
            database=f"error: {str(e)}",
            timestamp=datetime.now()
        )


@app.get("/api/alerts", response_model=List[Alert], tags=["Alerts"])
async def get_alerts(
    severity: Optional[SeverityEnum] = Query(None, description="Filter by severity"),
    alert_type: Optional[AlertTypeEnum] = Query(None, description="Filter by alert type"),
    peer_addr: Optional[str] = Query(None, description="Filter by peer address"),
    start_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination")
):
    """
    Query alerts with optional filters.
    
    Examples:
    - GET /api/alerts?severity=critical
    - GET /api/alerts?peer_addr=203.0.113.1
    - GET /api/alerts?alert_type=rpki_invalid&limit=50
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build query with filters
    query = "SELECT * FROM alerts WHERE 1=1"
    params = []
    
    if severity:
        query += " AND severity = %s"
        params.append(severity.value)
    
    if alert_type:
        query += " AND alert_type = %s"
        params.append(alert_type.value)
    
    if peer_addr:
        query += " AND peer_addr = %s"
        params.append(peer_addr)
    
    if start_time:
        query += " AND created_at >= %s"
        params.append(start_time)
    
    if end_time:
        query += " AND created_at <= %s"
        params.append(end_time)
    
    query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [Alert(**row) for row in results]


@app.get("/api/alerts/{alert_id}", response_model=Alert, tags=["Alerts"])
async def get_alert_by_id(
    alert_id: int = Path(..., description="Alert ID")
):
    """Get a specific alert by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return Alert(**result)


@app.get("/api/detections", response_model=List[Detection], tags=["Detections"])
async def get_detections(
    peer_addr: Optional[str] = Query(None, description="Filter by peer address"),
    anomaly_only: bool = Query(False, description="Return only anomalies"),
    start_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination")
):
    """
    Query ML detection results with optional filters.
    
    Examples:
    - GET /api/detections?anomaly_only=true
    - GET /api/detections?peer_addr=203.0.113.1&limit=50
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build query
    query = "SELECT * FROM ml_results WHERE 1=1"
    params = []
    
    if peer_addr:
        query += " AND peer_addr = %s"
        params.append(peer_addr)
    
    if anomaly_only:
        query += " AND ensemble_is_anomaly = TRUE"
    
    if start_time:
        query += " AND timestamp >= %s"
        params.append(start_time)
    
    if end_time:
        query += " AND timestamp <= %s"
        params.append(end_time)
    
    query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [Detection(**row) for row in results]


@app.get("/api/statistics", response_model=Statistics, tags=["Statistics"])
async def get_statistics(
    start_time: Optional[datetime] = Query(None, description="Start time for statistics"),
    end_time: Optional[datetime] = Query(None, description="End time for statistics")
):
    """
    Get system-wide statistics.
    
    Returns total alerts, detections, anomaly rates, and active peers.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build time filter
    time_filter = ""
    params = []
    
    if start_time:
        time_filter += " AND created_at >= %s"
        params.append(start_time)
    
    if end_time:
        time_filter += " AND created_at <= %s"
        params.append(end_time)
    
    # Total alerts
    cursor.execute(f"SELECT COUNT(*) FROM alerts WHERE 1=1{time_filter}", params)
    total_alerts = cursor.fetchone()['count']
    
    # Alerts by severity
    cursor.execute(f"""
        SELECT severity, COUNT(*) as count
        FROM alerts
        WHERE 1=1{time_filter}
        GROUP BY severity
    """, params)
    alerts_by_severity = {row['severity']: row['count'] for row in cursor.fetchall()}
    
    # Total detections
    time_filter_det = time_filter.replace('created_at', 'timestamp')
    cursor.execute(f"SELECT COUNT(*) FROM ml_results WHERE 1=1{time_filter_det}", params)
    total_detections = cursor.fetchone()['count']
    
    # Anomaly count
    cursor.execute(f"""
        SELECT COUNT(*) FROM ml_results 
        WHERE ensemble_is_anomaly = TRUE{time_filter_det}
    """, params)
    anomaly_count = cursor.fetchone()['count']
    
    # Anomaly rate
    anomaly_rate = (anomaly_count / total_detections * 100) if total_detections > 0 else 0.0
    
    # Active peers
    cursor.execute(f"""
        SELECT COUNT(DISTINCT peer_addr) FROM ml_results 
        WHERE 1=1{time_filter_det}
    """, params)
    active_peers = cursor.fetchone()['count']
    
    # Time range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ml_results")
    time_range = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return Statistics(
        total_alerts=total_alerts,
        alerts_by_severity=alerts_by_severity,
        total_detections=total_detections,
        anomaly_rate=round(anomaly_rate, 2),
        active_peers=active_peers,
        time_range={
            "start": str(time_range['min']) if time_range['min'] else "",
            "end": str(time_range['max']) if time_range['max'] else ""
        }
    )


@app.get("/api/peers", response_model=List[PeerInfo], tags=["Peers"])
async def get_peers(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
):
    """
    Get information about monitored BGP peers.
    
    Returns peer addresses with detection counts and anomaly rates.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            peer_addr,
            COUNT(*) as total_detections,
            SUM(CASE WHEN ensemble_is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
            MAX(timestamp) as last_seen
        FROM ml_results
        GROUP BY peer_addr
        ORDER BY total_detections DESC
        LIMIT %s
    """, (limit,))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    peers = []
    for row in results:
        anomaly_rate = (row['anomaly_count'] / row['total_detections'] * 100) if row['total_detections'] > 0 else 0.0
        peers.append(PeerInfo(
            peer_addr=row['peer_addr'],
            total_detections=row['total_detections'],
            anomaly_count=row['anomaly_count'],
            anomaly_rate=round(anomaly_rate, 2),
            last_seen=row['last_seen']
        ))
    
    return peers


# ============================================================================
# STARTUP/SHUTDOWN EVENTS (Handled by lifespan above)
# ============================================================================

@app.get("/api/live-stats", tags=["Statistics"])
async def get_live_stats():
    """
    Get live processing statistics for dashboard.
    Shows data processing even when no anomalies detected.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get quick counts from recent data (limit scan for performance)
    cursor.execute("""
        WITH recent_data AS (
            SELECT 
                ensemble_is_anomaly,
                heuristic_score,
                heuristic_is_anomaly,
                lstm_anomaly_score,
                lstm_is_anomaly,
                if_anomaly_score,
                if_is_anomaly,
                timestamp
            FROM ml_results
            ORDER BY id DESC
            LIMIT 500
        )
        SELECT 
            COUNT(*) as total_processed,
            COUNT(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 END) as anomalies,
            COUNT(CASE WHEN ensemble_is_anomaly = FALSE THEN 1 END) as normal,
            MAX(timestamp) as latest_processing,
            COUNT(CASE WHEN heuristic_score IS NOT NULL THEN 1 END) as heuristic_total,
            COUNT(CASE WHEN heuristic_is_anomaly = TRUE THEN 1 END) as heuristic_anomalies,
            COUNT(CASE WHEN lstm_anomaly_score IS NOT NULL THEN 1 END) as lstm_total,
            COUNT(CASE WHEN lstm_is_anomaly = TRUE THEN 1 END) as lstm_anomalies,
            COUNT(CASE WHEN if_anomaly_score IS NOT NULL THEN 1 END) as if_total,
            COUNT(CASE WHEN if_is_anomaly = TRUE THEN 1 END) as if_anomalies
        FROM recent_data;
    """)
    result = cursor.fetchone()
    
    processing = {
        'total_processed': result['total_processed'],
        'anomalies': result['anomalies'],
        'normal': result['normal'],
        'latest_processing': result['latest_processing']
    }
    
    models = {
        'heuristic_total': result['heuristic_total'],
        'heuristic_anomalies': result['heuristic_anomalies'],
        'lstm_total': result['lstm_total'],
        'lstm_anomalies': result['lstm_anomalies'],
        'if_total': result['if_total'],
        'if_anomalies': result['if_anomalies']
    }
    
    # Simple rate calculation
    rate = {'rate': result['total_processed'] / 5.0 if result['total_processed'] > 0 else 0}
    
    cursor.close()
    conn.close()
    
    return {
        "processing": {
            "total_processed": processing['total_processed'],
            "anomalies": processing['anomalies'],
            "normal": processing['normal'],
            "latest_processing": str(processing['latest_processing']) if processing['latest_processing'] else None
        },
        "models": {
            "heuristic": {
                "total": models['heuristic_total'],
                "anomalies": models['heuristic_anomalies'],
                "normal": models['heuristic_total'] - models['heuristic_anomalies']
            },
            "lstm": {
                "total": models['lstm_total'],
                "anomalies": models['lstm_anomalies'],
                "normal": models['lstm_total'] - models['lstm_anomalies']
            },
            "isolation_forest": {
                "total": models['if_total'],
                "anomalies": models['if_anomalies'],
                "normal": models['if_total'] - models['if_anomalies']
            }
        },
        "rate_per_minute": round(rate['rate'], 2)
    }


@app.get("/api/time-series", tags=["Statistics"])
async def get_time_series(
    hours: int = Query(1, description="Number of hours to look back", ge=1, le=24)
):
    """
    Get time-series data for anomaly scores over time.
    Returns data for graphing even when all scores are low (normal traffic).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Use LIMIT instead of time-based query for performance
    cursor.execute("""
        WITH recent_records AS (
            SELECT timestamp, ensemble_score, ensemble_is_anomaly
            FROM ml_results
            ORDER BY id DESC
            LIMIT 100
        )
        SELECT 
            DATE_TRUNC('minute', timestamp) as time_bucket,
            AVG(ensemble_score) as avg_score,
            MAX(ensemble_score) as max_score,
            COUNT(*) as count,
            COUNT(CASE WHEN ensemble_is_anomaly = TRUE THEN 1 END) as anomaly_count
        FROM recent_records
        GROUP BY time_bucket
        ORDER BY time_bucket DESC
        LIMIT 60;
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {
        "data": [
            {
                "timestamp": str(row['time_bucket']),
                "avg_score": float(row['avg_score']) if row['avg_score'] else 0,
                "max_score": float(row['max_score']) if row['max_score'] else 0,
                "count": row['count'],
                "anomaly_count": row['anomaly_count']
            }
            for row in results
        ]
    }


@app.get("/api/recent-detections", tags=["Detections"])
async def get_recent_detections(limit: int = Query(20, ge=1, le=100)):
    """
    Get recent detections (all, not just anomalies) for live monitoring table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            timestamp,
            peer_addr,
            heuristic_is_anomaly,
            heuristic_score,
            lstm_is_anomaly,
            lstm_anomaly_score,
            if_is_anomaly,
            if_anomaly_score,
            ensemble_is_anomaly,
            ensemble_score
        FROM ml_results
        ORDER BY timestamp DESC
        LIMIT %s;
    """, (limit,))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {
        "detections": [
            {
                "timestamp": str(row['timestamp']),
                "peer": row['peer_addr'][:30] if row['peer_addr'] else 'N/A',
                "heuristic": {
                    "anomaly": row['heuristic_is_anomaly'],
                    "score": float(row['heuristic_score']) if row['heuristic_score'] else None
                },
                "lstm": {
                    "anomaly": row['lstm_is_anomaly'],
                    "score": float(row['lstm_anomaly_score']) if row['lstm_anomaly_score'] else None
                },
                "isolation_forest": {
                    "anomaly": row['if_is_anomaly'],
                    "score": float(row['if_anomaly_score']) if row['if_anomaly_score'] else None
                },
                "ensemble": {
                    "anomaly": row['ensemble_is_anomaly'],
                    "score": float(row['ensemble_score']) if row['ensemble_score'] else 0
                }
            }
            for row in results
        ]
    }


@app.get("/api/drift-status", tags=["Monitoring"])
async def get_drift_status():
    """
    Get model drift monitoring status and recent retraining history.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get active drift flags
    cursor.execute("""
        SELECT 
            model,
            drift_detected,
            drift_score,
            threshold,
            timestamp,
            details
        FROM drift_flags
        WHERE timestamp > NOW() - INTERVAL '7 days'
        ORDER BY timestamp DESC
        LIMIT 10;
    """)
    flags = cursor.fetchall()
    
    # Get recent drift reports
    cursor.execute("""
        SELECT 
            model,
            timestamp,
            status,
            action_taken,
            metrics
        FROM drift_reports
        ORDER BY timestamp DESC
        LIMIT 5;
    """)
    reports = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return {
        "drift_flags": [
            {
                "model": row['model'],
                "drift_detected": row['drift_detected'],
                "drift_score": row['drift_score'],
                "threshold": row['threshold'],
                "timestamp": str(row['timestamp']),
                "details": row['details']
            }
            for row in flags
        ],
        "recent_reports": [
            {
                "model": row['model'],
                "timestamp": str(row['timestamp']),
                "status": row['status'],
                "action_taken": row['action_taken'],
                "metrics": row['metrics']
            }
            for row in reports
        ]
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
