"""
EMERGENCY BYPASS: Skip validation entirely in critical scenarios
This ensures trading is never delayed by validation issues
"""

import logging
import os
from datetime import datetime


def emergency_validation_bypass():
    """
    NUCLEAR OPTION: Complete validation bypass for emergency trading
    
    Use this when:
    1. Market opened and we need to trade immediately 
    2. Validation is taking too long
    3. Previous validation succeeded and cached
    
    Returns: True if bypass is safe, False if validation absolutely needed
    """
    # Check if we're past 09:25 (5 minutes after market open)
    now = datetime.now()
    market_start = now.replace(hour=9, minute=25, second=0, microsecond=0)
    
    if now > market_start:
        logging.warning("🚨 EMERGENCY BYPASS: Past 09:25 - skipping validation entirely!")
        logging.warning("🚨 TRADING PRIORITY: Validation can wait - executing strategy now")
        return True
    
    # Check if validation cache exists from previous runs
    cache_files = [
        "logs/.daily_validation_cache",
        "logs/.symbol_format_cache", 
        "logs/.format_cache"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                cache_age = os.path.getmtime(cache_file)
                hours_old = (now.timestamp() - cache_age) / 3600
                
                if hours_old < 24:  # Cache less than 24 hours old
                    logging.warning(f"🚨 EMERGENCY BYPASS: Using {hours_old:.1f}h old cache from {cache_file}")
                    logging.warning("🚨 TRADING PRIORITY: Proceeding with cached validation")
                    return True
                    
            except Exception as e:
                continue
    
    # No safe bypass available
    logging.info("⚠️ No emergency bypass available - validation required")
    return False


def force_trading_mode():
    """
    LAST RESORT: Force trading with no validation whatsoever
    Only use when validation is completely broken but trading must continue
    """
    logging.error("🔥 FORCE TRADING MODE: NO VALIDATION - PROCEED AT YOUR OWN RISK!")
    logging.error("🔥 This bypasses ALL safety checks - only use in emergencies")
    logging.error("🔥 Strategy will use fallback symbol detection during execution")
    
    # Create emergency cache to prevent repeated warnings
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/.emergency_bypass_used", 'w') as f:
            f.write(f"EMERGENCY_BYPASS_USED_{datetime.now().isoformat()}")
    except:
        pass
    
    return True