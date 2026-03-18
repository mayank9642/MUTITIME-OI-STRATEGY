"""
Test the trailing stop loss implementation using the fixed strategy file
"""
import sys
import os
import importlib.util

# Add the directory containing the fixed strategy to the Python path
sys.path.append('c:\\vs code projects\\finalized strategies\\src')

# Load the module directly from file path
spec = importlib.util.spec_from_file_location("strategy_fixed", 
                                            "c:\\vs code projects\\finalized strategies\\src\\strategy_fixed.py")
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)

# Get the OpenInterestStrategy class
OpenInterestStrategy = strategy_module.OpenInterestStrategy

print("=== Testing Trailing Stop Loss with Fixed Strategy File ===")

# Create a strategy instance
strategy = OpenInterestStrategy()

# Create a simulated trade
strategy.active_trade = {
    'symbol': 'NIFTY25JUL19000CE',
    'entry_price': 100.0,
    'quantity': 25,
    'stoploss': 90.0,
    'target': 120.0,
}

# Test the trailing stop loss functionality with different prices
test_prices = [105, 110, 115]

for price in test_prices:
    print(f"\nTesting with price: {price}")
    result = strategy.update_trailing_stoploss(price)
    print(f"Update successful: {result}")
    print(f"Current stop loss: {strategy.active_trade['stoploss']}")

print("\nFinal stop loss:", strategy.active_trade['stoploss'])
print("Original stop loss:", strategy.active_trade.get('original_stoploss', 'Not set'))

if strategy.active_trade['stoploss'] > 90.0:
    print("\n✓ PASSED: Trailing stop loss is working correctly in the fixed strategy file!")
else:
    print("\n✗ FAILED: Trailing stop loss did not update correctly!")
