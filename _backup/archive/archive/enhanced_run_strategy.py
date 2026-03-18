"""
Enhanced strategy runner that uses the FixedOpenInterestStrategy
to attempt to take a trade without requiring update_trailing_stoploss method
"""
import os
import sys
import logging
import datetime
import time
import pytz

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/enhanced_run.log')
    ]
)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("ENHANCED STRATEGY EXECUTION")
print("=" * 80)

try:
    # Import the fixed strategy class
    from src.fixed_strategy_updated import FixedOpenInterestStrategy
    
    # Create strategy instance
    strategy = FixedOpenInterestStrategy()
    print(f"Strategy instance created: {type(strategy).__name__}")
    
    # Initialize the strategy
    print("Initializing strategy...")
    init_result = strategy.initialize_day()
    
    if not init_result:
        print("Failed to initialize strategy. Exiting.")
        sys.exit(1)
    
    print("Strategy initialized successfully.")
    
    # Get current time in IST
    ist_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    print(f"Current IST time: {ist_time}")
    
    # Run the strategy with force_analysis=True to attempt to take a trade
    print("Running strategy with force_analysis=True to take a trade...")
    
    # Connect to the websocket for real-time data if available
    print("Setting up real-time data feed...")
    try:
        if hasattr(strategy, 'start_data_websocket'):
            strategy.start_data_websocket()
            print("Real-time data feed established")
        else:
            print("No websocket method available, continuing without real-time data")
    except Exception as e:
        print(f"Error setting up data feed: {e}")
    
    # Execute the strategy
    run_result = strategy.run_strategy(force_analysis=True)
    print(f"Strategy run result: {run_result}")
    
    # Monitor for some time to see if a trade is taken
    print("Monitoring for trades for 30 seconds...")
    start_time = time.time()
    while time.time() - start_time < 30:
        if hasattr(strategy, 'active_trade') and strategy.active_trade:
            print(f"Trade detected: {strategy.active_trade}")
            break
        time.sleep(2)
        print(".", end="", flush=True)
    
    print("\nStrategy execution completed.")
    
    # If a trade was taken, show details
    if hasattr(strategy, 'active_trade') and strategy.active_trade:
        print("\nTrade details:")
        for key, value in strategy.active_trade.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo trade was taken during the monitoring period.")
    
    print("\nStrategy run completed!")
    
except Exception as e:
    import traceback
    print(f"Error running strategy: {e}")
    print(traceback.format_exc())

print("=" * 80)
