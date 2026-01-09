# 🛡️ BGP Anomaly Detection System - Complete Setup Guide

## ✅ All Problems Fixed!

Your BGP monitoring system now has:
- ✅ **Limited RIS Live streaming** (no infinite data)
- ✅ **Controlled test data** (stops automatically)
- ✅ **Database cleanup** (auto-delete old records)
- ✅ **Memory monitoring** (prevents CPU overload)
- ✅ **Rate limiting** (protects system resources)
- ✅ **Manual controls** (API endpoints to stop services)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd routinator
pip install fastapi uvicorn websockets httpx sqlalchemy psycopg2-binary python-dotenv psutil
```

### 2. Setup Database (One-time)
```bash
# Apply database triggers for automatic cleanup
psql -U postgres -d bgp_monitor -f ../sql/database_triggers.sql
```

### 3. Configure Limits (Optional)
Edit [`config/system_limits.json`](config/system_limits.json ) to adjust limits:
```json
{
  "ris_live": {
    "max_messages": 10000,        // Stop after 10K messages
    "max_runtime_hours": 24       // Stop after 24 hours
  },
  "database": {
    "max_bgp_announcements": 50000  // Max 50K records
  }
}
```

### 4. Start the System
```bash
cd routinator
python run.py
```

You'll see:
```
╔══════════════════════════════════════════════════════════════╗
║          BGP Anomaly Detection System v2.0                   ║
║                                                              ║
║  ✓ RIS Live WebSocket streaming (with limits)               ║
║  ✓ Database with automatic cleanup                           ║
║  ✓ Memory monitoring and protection                          ║
║                                                              ║
║  API Docs: http://localhost:8000/docs                        ║
╚══════════════════════════════════════════════════════════════╝
```

### 5. Monitor the System
```bash
# Open in browser:
http://localhost:8000/docs

# Or use curl:
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/system/health
```

---

## 📊 Monitoring & Control

### Check System Health
```bash
curl http://localhost:8000/api/system/health
```

Response:
```json
{
  "status": "healthy",
  "memory_mb": 245.6,
  "database_records": 1523,
  "database_limit": 50000,
  "database_usage_percent": 3.05,
  "ris_live_running": true
}
```

### View Statistics
```bash
curl http://localhost:8000/api/stats
```

Response:
```json
{
  "total": 1523,
  "valid": 892,
  "invalid": 45,
  "not_found": 586,
  "ris_live": {
    "message_count": 1523,
    "max_messages": 10000,
    "runtime_seconds": 1245.3,
    "running": true
  }
}
```

### Stop Services Manually

**Stop RIS Live:**
```bash
curl -X POST http://localhost:8000/api/control/stop_ris_live
```

**Stop Test Data Generator:**
```bash
curl -X POST http://localhost:8000/api/control/stop_test_data
```

**Trigger Database Cleanup:**
```bash
curl -X POST http://localhost:8000/api/database/cleanup
```

---

## 🔧 Configuration Reference

### [`config/system_limits.json`](config/system_limits.json )

| Setting | Default | Description |
|---------|---------|-------------|
| `ris_live.max_messages` | 10000 | Stop RIS Live after this many messages |
| `ris_live.max_runtime_hours` | 24 | Stop RIS Live after this many hours |
| `ris_live.message_rate_limit_per_second` | 100 | Max messages per second |
| `database.max_bgp_announcements` | 50000 | Max records in database |
| `database.cleanup_older_than_days` | 30 | Delete records older than this |
| `database.auto_cleanup_enabled` | true | Enable automatic cleanup |
| `database.cleanup_interval_hours` | 6 | Run cleanup every N hours |
| `test_data_generator.max_records` | 1000 | Max test records to generate |
| `test_data_generator.auto_stop_after_hours` | 1 | Stop test generator after N hours |
| `memory.max_memory_mb` | 2048 | Critical memory limit (2GB) |
| `memory.warning_threshold_mb` | 1536 | Warning threshold (1.5GB) |

---

## 🏗️ Architecture Overview

```
Layer 0: Network Infrastructure
  Router → RIS Live Client (WebSocket)

Layer 1: Data Ingestion
  RIS Live Collector → PostgreSQL
  ├─ With limits (10K messages, 24h runtime)
  ├─ Rate limiting (100 msg/sec)
  └─ Auto-cleanup (keep 50K records)

Layer 2: Storage & Feature Processing
  PostgreSQL Database
  ├─ bgp_announcements (max 50K records)
  ├─ Auto-cleanup triggers
  └─ Monitoring views

Layer 3: Detection & Correlation
  [Next phase - ML inference services]
  ├─ LSTM Autoencoder
  ├─ Isolation Forest
  └─ RPKI Validator

Layer 4: Presentation & Integration
  FastAPI Backend
  ├─ REST API
  ├─ WebSocket streaming
  └─ Health monitoring
```

---

## 📁 File Structure

```
lstm_model/
├── config/
│   └── system_limits.json          # All configurable limits ✨NEW
│
├── routinator/
│   ├── main.py                     # FastAPI app with limits ✨UPDATED
│   ├── ris_live_client.py          # RIS Live with limits ✨NEW
│   ├── database.py                 # DB with cleanup functions ✨UPDATED
│   ├── routinator_client.py        # RPKI validation client
│   └── run.py                      # Start script ✨NEW
│
├── sql/
│   └── database_triggers.sql       # PostgreSQL triggers ✨NEW
│
├── model_output/                   # Trained ML models
│   ├── lstm/
│   └── isolation_forest.pkl
│
└── FIXES_APPLIED.md                # Detailed fix documentation ✨NEW
```

---

## 🔍 Database Queries

### View Current Statistics
```sql
SELECT * FROM bgp_monitoring_stats;
```

### Get Table Size
```sql
SELECT * FROM get_table_info('bgp_announcements');
```

### Manual Cleanup (keep last 30 days)
```sql
SELECT * FROM cleanup_old_bgp_data(30);
```

### Recent Invalid Routes
```sql
SELECT * FROM bgp_announcements 
WHERE status = 'invalid' 
  AND timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/api/stats` | BGP statistics |
| `GET` | `/api/system/health` | System health check |
| `GET` | `/api/announcements` | Recent announcements |
| `GET` | `/api/validate/{asn}/{prefix}` | Validate ASN/prefix |
| `POST` | `/api/control/stop_ris_live` | Stop RIS Live |
| `POST` | `/api/control/stop_test_data` | Stop test generator |
| `POST` | `/api/database/cleanup` | Manual cleanup |
| `WS` | `/ws` | WebSocket real-time updates |

---

## 🧪 Testing

### Test RIS Live Limits
1. Start the system: `python run.py`
2. Watch logs for message count
3. Verify it stops at 10,000 messages or 24 hours

### Test Database Cleanup
```python
# In Python shell
from routinator.database import SessionLocal, cleanup_old_records

db = SessionLocal()
deleted = cleanup_old_records(db, keep_last=1000)
print(f"Deleted {deleted} records")
```

### Test Memory Monitoring
```bash
# Monitor memory usage
watch -n 5 'curl -s http://localhost:8000/api/system/health | jq ".memory_mb"'
```

---

## 🛟 Troubleshooting

### Problem: RIS Live not connecting
**Solution**: 
- Check internet connection
- Verify RIPE RIS Live is accessible: `curl https://ris-live.ripe.net`
- Check firewall settings

### Problem: Database errors
**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U postgres -d bgp_monitor -c "SELECT COUNT(*) FROM bgp_announcements;"
```

### Problem: High memory usage
**Solution**:
- Check `/api/system/health` for memory usage
- Reduce `max_bgp_announcements` in config
- Trigger manual cleanup: `POST /api/database/cleanup`

### Problem: RIS Live runs too long
**Solution**:
- Reduce `max_runtime_hours` in [`config/system_limits.json`](config/system_limits.json )
- Stop manually: `POST /api/control/stop_ris_live`

---

## 🎯 Production Deployment

### 1. Disable Test Data Generator
```json
// config/system_limits.json
{
  "test_data_generator": {
    "enabled": false    // Set to false
  }
}
```

### 2. Adjust Limits for Production
```json
{
  "ris_live": {
    "max_messages": 100000,        // Higher limit
    "max_runtime_hours": 168       // 1 week
  },
  "database": {
    "max_bgp_announcements": 500000  // 500K records
  }
}
```

### 3. Enable Database Triggers
```bash
psql -U postgres -d bgp_monitor -f sql/database_triggers.sql
```

### 4. Setup Monitoring
- Monitor `/api/system/health` every 5 minutes
- Alert if `database_usage_percent > 90`
- Alert if `memory_mb > 1800`
- Alert if `ris_live_running = false` unexpectedly

### 5. Run as Service (systemd)
```ini
# /etc/systemd/system/bgp-monitor.service
[Unit]
Description=BGP Anomaly Detection Service
After=network.target postgresql.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/lstm_model/routinator
ExecStart=/path/to/venv/bin/python run.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable bgp-monitor
sudo systemctl start bgp-monitor
sudo systemctl status bgp-monitor
```

---

## 📚 Next Steps

1. ✅ **Current**: Data ingestion with limits (Layer 0-2)
2. 🔄 **Next**: Add ML inference services (Layer 3)
   - Load LSTM and Isolation Forest models
   - Process batches of BGP data
   - Generate anomaly scores
3. 🔄 **Future**: Add presentation layer (Layer 4)
   - Web dashboard
   - Alert management
   - ITSM integration

---

## 🆘 Support

For issues or questions:
1. Check logs: `tail -f /var/log/bgp-monitor.log`
2. Review [`FIXES_APPLIED.md`](FIXES_APPLIED.md ) for detailed fix documentation
3. Test endpoints using Swagger UI: `http://localhost:8000/docs`

---

**System Status**: ✅ Production Ready
**Version**: 2.0.0
**Last Updated**: January 8, 2026
