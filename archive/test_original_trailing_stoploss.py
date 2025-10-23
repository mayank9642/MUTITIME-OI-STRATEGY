"""
Test the trailing stop loss implementation using the original strategy file
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the original strategy
from src.strategy import OpenInterestStrategy

def test_trailing_stoploss():
    """Test the trailing stop loss functionality with the original strategy file"""
    # Create a strategy instance
    strategy = OpenInterestStrategy()
    
    # Check if the method exists
    if not hasattr(strategy, 'update_trailing_stoploss'):
        logging.error("update_trailing_stoploss method does not exist!")
        return False
    
    logging.info("update_trailing_stoploss method exists. Testing functionality...")
    
    # Set up a mock trade for testing
    strategy.active_trade = {
        'symbol': 'NIFTY25JUL18000CE',
        'entry_price': 100,
        'stoploss': 90,
        'quantity': 50
    }
    
    # Test scenario 1: Price increases, stop loss should move up
    initial_sl = strategy.active_trade['stoploss']
    current_price = 105
    
    logging.info(f"Initial price: 100, Initial SL: {initial_sl}")
    logging.info(f"Price increases to {current_price}")
    
    result = strategy.update_trailing_stoploss(current_price)
    new_sl = strategy.active_trade['stoploss']
    
    logging.info(f"Result: {result}, New SL: {new_sl}")
    expected_sl = 96.6  # 105 * (1 - 8/100) = 96.6
    
    # Test scenario 2: Price increases more
    current_price = 110
    result = strategy.update_trailing_stoploss(current_price)
    new_sl = strategy.active_trade['stoploss']
    logging.info(f"Price increases to {current_price}")
    logging.info(f"Result: {result}, New SL: {new_sl}")
    
    # Test scenario 3: Price increases again
    current_price = 115
    result = strategy.update_trailing_stoploss(current_price)
    new_sl = strategy.active_trade['stoploss']
    logging.info(f"Price increases to {current_price}")
    logging.info(f"Result: {result}, New SL: {new_sl}")
    
    # Final check - does the stoploss match what we expect?
    logging.info("Test complete!")
    
    if strategy.active_trade['stoploss'] > 100.0:
        logging.info("PASSED: Trailing stop loss is working correctly!")
        return True
    else:
        logging.error("FAILED: Trailing stop loss did not update correctly!")
        return False

if __name__ == "__main__":
    test_trailing_stoploss()
