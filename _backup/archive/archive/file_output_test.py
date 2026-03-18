"""
Direct test that writes results to a file
"""
import os

# Create output directory
os.makedirs("test_results", exist_ok=True)

# Open output file
with open("test_results/trailing_sl_test.txt", "w") as f:
    f.write("=== DIRECT TEST OF TRAILING STOP LOSS ===\n")

    try:
        # Access the method directly from strategy.py
        from src.strategy import OpenInterestStrategy
        f.write("Successfully imported OpenInterestStrategy\n")

        # Create an instance
        strategy = OpenInterestStrategy()
        f.write("Successfully created strategy instance\n")

        # Set up mock trade
        strategy.active_trade = {
            'symbol': 'NIFTY25JUL18000CE',
            'entry_price': 150.0,
            'stoploss': 130.0
        }
        f.write(f"Mock trade: {strategy.active_trade}\n")

        # Test with price movements
        test_prices = [160.0, 170.0, 150.0, 180.0]
        for price in test_prices:
            f.write(f"\nTesting with price: {price}\n")
            result = strategy.update_trailing_stoploss(price)
            f.write(f"Update result: {result}\n")
            f.write(f"Current stoploss: {strategy.active_trade.get('stoploss')}\n")

        f.write("\n=== TEST COMPLETED SUCCESSFULLY ===\n")
    except Exception as e:
        import traceback
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
