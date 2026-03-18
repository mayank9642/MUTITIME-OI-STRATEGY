"""
Symbol Format Validator
Tests which symbol format Fyers is currently accepting to prevent trading failures
"""
import logging
import re
from datetime import datetime
import pytz
import time
import traceback
from src.fetch_option_oi import fetch_option_oi
from src.symbol_utils import convert_option_symbol_format
from src.fixed_improved_websocket import improved_market_data_websocket

# Known-good fallback symbols to try when NSE option-chain fetch fails
# Updated with current December 2025 D-code format (25D05 for Dec 5, 25D09 for Dec 9)
FALLBACK_TEST_SYMBOLS = [
    "NSE:NIFTY25D0526000CE",  # December 5, 2025 D-code format (today)
    "NSE:NIFTY25D0526100CE",  # Alternative December 5 strike
    "NSE:NIFTY25D0926000CE",  # December 9, 2025 D-code format (known working)
    "NSE:NIFTY25D0926100CE",  # Alternative December 9 strike
    "NSE:NIFTY25NOV26000CE"   # Full month name style (fallback)
]
def validate_symbol_format():
    """
    Test the current symbol format by attempting a websocket connection
    Returns the working format: 'full' (NOV), 'abbreviated' (N) , 'dcode' (YYD) or None if all fail
    """
    logging.info("=" * 60)
    logging.info("SYMBOL FORMAT VALIDATION - Testing current Fyers format")
    logging.info("=" * 60)

    try:
        # Get a sample option symbol from OI data
        logging.info("Fetching current option chain to get test symbols...")
        oi_data = fetch_option_oi('NIFTY')

        test_symbols = []
        if oi_data is not None and len(oi_data) > 0:
            logging.info("FORCING D-CODE SYMBOL SELECTION (ignoring Fyers format chaos)")
            
            # IGNORE EXPIRY - Just find ANY working D-code symbol from the entire option chain
            all_ce_options = oi_data[oi_data['option_type'] == 'CE'].sort_values('strike')
            
            # Look for D-code symbols across ALL expiries (25D pattern)
            dcode_symbols = all_ce_options[all_ce_options['symbol'].str.contains('25D', na=False)]
            
            if not dcode_symbols.empty:
                # Pick a middle D-code symbol for testing  
                mid_index = len(dcode_symbols) // 2
                selected_symbol = dcode_symbols.iloc[mid_index]['symbol']
                test_symbols = [selected_symbol]
                
                # Get the expiry for this symbol for logging
                selected_expiry = dcode_symbols.iloc[mid_index]['expiry']
                
                logging.info(f"✓ FOUND D-CODE SYMBOL: {selected_symbol}")
                logging.info(f"✓ From expiry: {selected_expiry}")
                logging.info(f"✓ Total D-code symbols available: {len(dcode_symbols)}")
            else:
                logging.error("❌ NO D-CODE SYMBOLS FOUND IN ENTIRE OPTION CHAIN!")
                logging.error("❌ Fyers may have changed format again - using fallback symbols")
                test_symbols = FALLBACK_TEST_SYMBOLS[:]
        else:
            logging.error("Could not fetch option chain for validation - using fallback symbols")
            test_symbols = FALLBACK_TEST_SYMBOLS[:]

        # Add fallback symbols to the test list if primary symbol fails
        all_test_symbols = test_symbols + FALLBACK_TEST_SYMBOLS
        # Remove duplicates while preserving order
        seen = set()
        unique_test_symbols = []
        for sym in all_test_symbols:
            if sym not in seen:
                unique_test_symbols.append(sym)
                seen.add(sym)
        
        # Try each candidate symbol until one validates
        for test_symbol in unique_test_symbols:
            logging.info(f"Testing websocket subscription with symbol: {test_symbol}")
            test_client = improved_market_data_websocket(
                symbols=[test_symbol],
                callback_handler=None,
                debug=False
            )

            if test_client is None:
                logging.error("Websocket connection failed - trying next fallback if available")
                continue

            # Wait briefly to check if subscription succeeds or errors
            time.sleep(2)

            result_type = 'unknown'
            if hasattr(test_client, 'connection_status'):
                status = test_client.connection_status
                subscribed = status.get('subscribed', False)
                sub_error = status.get('subscription_error')
                connected = status.get('connected', False)

                # Consider success if we are subscribed (confirmed by tick)
                # OR connected without a subscription error after the wait
                if subscribed is True or (connected and not sub_error and subscribed in (True, 'pending')):
                    if re.search(r"NSE:[A-Z]+\d{2}D\d{2}", test_symbol):
                        result_type = 'dcode'
                        logging.info("✓ Symbol format validation PASSED: D-code December/weekly format is working")
                    elif any(m in test_symbol for m in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
                        result_type = 'full'
                        logging.info("✓ Symbol format validation PASSED: Full month names (NOV, DEC, etc.)")
                    else:
                        result_type = 'abbreviated'
                        logging.info("✓ Symbol format validation PASSED: Abbreviated month codes (N, D, etc.)")

                    if hasattr(test_client, 'close_connection'):
                        test_client.close_connection()

                    logging.info("=" * 60)
                    logging.info(f"VALIDATION RESULT: {result_type.upper()} format is working")
                    logging.info("=" * 60)
                    return result_type
                else:
                    logging.error("Websocket connected but subscription not confirmed; will try next symbol if any")
                    logging.error(f"Connection status: {status}")
                    if hasattr(test_client, 'close_connection'):
                        test_client.close_connection()
                    continue
            else:
                logging.warning("Could not verify subscription status; trying next symbol if any")
                if hasattr(test_client, 'close_connection'):
                    test_client.close_connection()
                continue

        # If all candidates failed
        logging.error("All symbol format validation attempts failed")
        return None

    except Exception as e:
        logging.error(f"Symbol validation failed with error: {e}")
        logging.error(traceback.format_exc())
        return None


def get_recommended_format():
    """
    Returns the recommended symbol format based on validation
    Returns: dict with 'type' and 'description'
    """
    format_type = validate_symbol_format()
    
    if format_type == 'full':
        return {
            'type': 'full',
            'description': 'Full month names (e.g., NSE:NIFTY25NOV26000CE)',
            'pattern': 'NIFTY + day + MONTH + strike + CE/PE'
        }
    elif format_type == 'abbreviated':
        return {
            'type': 'abbreviated',
            'description': 'Abbreviated month codes (e.g., NSE:NIFTY25N26000CE)',
            'pattern': 'NIFTY + day + M + strike + CE/PE'
        }
    else:
        return {
            'type': 'unknown',
            'description': 'Could not determine working format - manual check required',
            'pattern': 'Unknown'
        }


# =====================================================================
# PERMANENT SOLUTION: Adaptive Format Detection Integration
# =====================================================================

def validate_symbol_format_permanent() -> str:
    """
    ULTIMATE SOLUTION: Just use Fyers' own symbols (no format guessing bullshit!)
    
    This function eliminates all format detection complexity by:
    1. First trying simple validation with Fyers' own symbols (SHOULD ALWAYS WORK)
    2. Fallback to current D-code method if needed
    3. Last resort: adaptive detection for edge cases
    
    Returns: Format type string or None
    """
    # SIMPLEST APPROACH: Use symbols Fyers gives us (should always work)
    logging.info("🎯 TRYING ULTIMATE SIMPLE APPROACH: Use Fyers' own symbols")
    
    try:
        from src.simple_fyers_validator import validate_using_fyers_own_symbols
        simple_result = validate_using_fyers_own_symbols()
        
        if simple_result == 'working':
            logging.info("✅ SIMPLE VALIDATION SUCCESS: Fyers symbols work!")
            logging.info("🎯 No format detection needed - using whatever Fyers gives us")
            return 'fyers_native'  # New format type meaning "use Fyers' own symbols"
    except ImportError:
        logging.warning("Simple validator not available")
    except Exception as e:
        logging.warning(f"Simple validation failed: {e}")
    
    # Fallback 1: Try current optimized D-code method
    logging.info("Fallback: Trying optimized D-code validation...")
    current_result = validate_symbol_format()
    
    if current_result and current_result != 'unknown':
        logging.info(f"✓ D-code fallback successful: {current_result}")
        return current_result
    
    # Fallback 2: Use adaptive detection (last resort)
    logging.warning("All simple methods failed, trying adaptive detection...")
    
    try:
        from src.adaptive_symbol_manager import validate_symbol_format_adaptive
        adaptive_result = validate_symbol_format_adaptive()
        
        if adaptive_result:
            logging.info(f"🎯 ADAPTIVE DETECTION SUCCESS: {adaptive_result}")
            return adaptive_result
        else:
            logging.error("❌ ALL VALIDATION METHODS FAILED")
            return None
            
    except ImportError:
        logging.error("Adaptive system not available - validation failed")
        return current_result


if __name__ == "__main__":
    # Test both legacy and adaptive validators
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("TESTING SYMBOL FORMAT VALIDATION")
    print("=" * 70)
    
    # Test current method
    print("\n1. Testing current optimized method:")
    result1 = validate_symbol_format()
    print(f"Result: {result1}")
    
    # Test permanent adaptive method
    print("\n2. Testing permanent adaptive method:")
    result2 = validate_symbol_format_permanent()
    print(f"Result: {result2}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION: Use validate_symbol_format_permanent() for future-proof validation")
    print("=" * 70)
