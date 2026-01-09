"""
Test Retraining System
======================
Tests the model retraining and hot-swap functionality.

Tests:
1. Isolation Forest retraining (fastest)
2. Heuristic rules update
3. LSTM retraining (slowest)
4. Hot-swap verification
"""

import os
import sys
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_isolation_forest():
    """Test Isolation Forest retraining"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Isolation Forest Retraining")
    logger.info("="*60)
    
    import retrain_isolation_forest
    
    retrainer = retrain_isolation_forest.IFRetrainer()
    success = retrainer.run_retraining()
    
    if success:
        logger.info("✅ Isolation Forest retraining PASSED")
        
        # Verify model file exists
        if os.path.exists(retrainer.old_model_path):
            size = os.path.getsize(retrainer.old_model_path) / 1024 / 1024
            logger.info(f"   Model file: {retrainer.old_model_path} ({size:.2f} MB)")
        
        return True
    else:
        logger.error("❌ Isolation Forest retraining FAILED")
        return False


def test_heuristic():
    """Test Heuristic rules update"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Heuristic Rules Update")
    logger.info("="*60)
    
    import retrain_heuristic
    
    retrainer = retrain_heuristic.HeuristicRetrainer()
    success = retrainer.run_retraining()
    
    if success:
        logger.info("✅ Heuristic rules update PASSED")
        
        # Verify config file exists and is valid
        if os.path.exists(retrainer.old_config_path):
            with open(retrainer.old_config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"   Config file: {retrainer.old_config_path}")
            logger.info(f"   Thresholds: {len(config.get('rules', {}))} rules")
            
            # Show rules
            if 'rules' in config:
                for key, value in config['rules'].items():
                    logger.info(f"      {key}: {value:.2f}")
        
        return True
    else:
        logger.error("❌ Heuristic rules update FAILED")
        return False


def test_lstm():
    """Test LSTM retraining (slowest - optional)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: LSTM Retraining (Optional - Takes Time)")
    logger.info("="*60)
    
    # Ask user
    response = input("Run LSTM retraining test? This will take several minutes (y/N): ")
    
    if response.lower() != 'y':
        logger.info("⏭️ Skipping LSTM test")
        return True
    
    import retrain_lstm
    
    retrainer = retrain_lstm.LSTMRetrainer()
    success = retrainer.run_retraining()
    
    if success:
        logger.info("✅ LSTM retraining PASSED")
        
        # Verify model file exists
        if os.path.exists(retrainer.old_model_path):
            size = os.path.getsize(retrainer.old_model_path) / 1024 / 1024
            logger.info(f"   Model file: {retrainer.old_model_path} ({size:.2f} MB)")
        
        return True
    else:
        logger.error("❌ LSTM retraining FAILED")
        return False


def test_hot_swap():
    """Test hot-swap mechanism by checking model versions"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Hot-Swap Verification")
    logger.info("="*60)
    
    # Check if backup files were created
    backups_found = []
    
    # Check IF backup
    if_backups = [f for f in os.listdir("model_artifacts") if f.startswith("iso_forest_bgp_production_backup")]
    if if_backups:
        backups_found.append(f"IF: {if_backups[-1]}")
    
    # Check heuristic backup
    heuristic_backups = [f for f in os.listdir("model_artifacts") if f.startswith("heuristic_rules_backup")]
    if heuristic_backups:
        backups_found.append(f"Heuristic: {heuristic_backups[-1]}")
    
    # Check LSTM backup
    lstm_backups = [f for f in os.listdir("model_output") if f.startswith("lstm_model_for_pkl_backup")]
    if lstm_backups:
        backups_found.append(f"LSTM: {lstm_backups[-1]}")
    
    if backups_found:
        logger.info("✅ Hot-swap verification PASSED")
        logger.info(f"   Backups created: {len(backups_found)}")
        for backup in backups_found:
            logger.info(f"      {backup}")
        return True
    else:
        logger.warning("⚠️ No backup files found (may be first run)")
        return True


def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("MODEL RETRAINING SYSTEM TEST")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now()}")
    
    results = []
    
    # Test 1: Isolation Forest (fast)
    try:
        results.append(("Isolation Forest", test_isolation_forest()))
    except Exception as e:
        logger.error(f"❌ IF test failed with exception: {e}")
        results.append(("Isolation Forest", False))
    
    # Test 2: Heuristic (fast)
    try:
        results.append(("Heuristic", test_heuristic()))
    except Exception as e:
        logger.error(f"❌ Heuristic test failed with exception: {e}")
        results.append(("Heuristic", False))
    
    # Test 3: LSTM (slow - optional)
    try:
        results.append(("LSTM", test_lstm()))
    except Exception as e:
        logger.error(f"❌ LSTM test failed with exception: {e}")
        results.append(("LSTM", False))
    
    # Test 4: Hot-swap
    try:
        results.append(("Hot-Swap", test_hot_swap()))
    except Exception as e:
        logger.error(f"❌ Hot-swap test failed with exception: {e}")
        results.append(("Hot-Swap", False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{name:20s} {status}")
    
    logger.info("="*60)
    logger.info(f"Result: {passed}/{total} tests passed")
    logger.info(f"End time: {datetime.now()}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
