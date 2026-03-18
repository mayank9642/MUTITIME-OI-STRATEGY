"""
ULTRA-FAST VALIDATION: Minimal startup check
Only validates if absolutely necessary, uses aggressive caching
"""

import os
import time
import logging
from datetime import datetime


def ultra_fast_validation_check():
    """
    Lightning-fast validation that skips heavy checks unless required.
    
    Strategy:
    1. Check if validation was done today
    2. If yes, skip entirely 
    3. If no, do minimal quick check only
    
    Returns: True if validation ok to skip, False if needs full validation
    """
    cache_file = "logs/.daily_validation_cache"
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_date, cached_format = f.read().strip().split('|')
            
            if cached_date == today_str:
                logging.info(f"✅ VALIDATION SKIPPED: Already validated today with format '{cached_format}'")
                logging.info("⚡ FAST STARTUP: No validation needed")
                return True
        
        # Need to validate - but cache result for entire day
        logging.info("🔍 FIRST RUN TODAY: Quick validation required")
        return False
        
    except Exception as e:
        logging.warning(f"Validation cache error: {e}")
        return False


def cache_validation_result(format_result):
    """Cache validation result for the entire trading day"""
    try:
        cache_file = "logs/.daily_validation_cache"
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        os.makedirs("logs", exist_ok=True)
        with open(cache_file, 'w') as f:
            f.write(f"{today_str}|{format_result}")
        
        logging.info(f"💾 Validation cached for entire day: {format_result}")
        
    except Exception as e:
        logging.warning(f"Cache write failed: {e}")


def minimal_symbol_test():
    """
    MINIMAL test - just verify one known symbol works
    No option chain fetch, no heavy processing
    """
    logging.info("⚡ MINIMAL VALIDATION: Testing one symbol only")
    
    try:
        # Test with a simple known symbol - no option chain needed
        from src.fixed_improved_websocket import enhanced_start_market_data_websocket
        
        # Use a basic index symbol that should always exist
        test_symbol = "NSE:NIFTY50-INDEX"
        
        logging.info(f"Testing basic symbol: {test_symbol}")
        
        ws = enhanced_start_market_data_websocket([test_symbol], lambda *args: None)
        
        if ws is not None:
            logging.info("✅ MINIMAL VALIDATION PASSED: Basic connectivity OK")
            try:
                ws.disconnect() if hasattr(ws, 'disconnect') else None
            except:
                pass
            return 'basic_connectivity'
        else:
            logging.warning("⚠️ Basic connectivity test failed")
            return None
            
    except Exception as e:
        logging.error(f"Minimal validation error: {e}")
        return None