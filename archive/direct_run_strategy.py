"""
Direct strategy runner that bypasses all monkey patching
and uses the FixedOpenInterestStrategy class directly
"""
import os
import sys
import logging
import datetime

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/direct_run.log')
    ]
)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("RUNNING STRATEGY DIRECTLY")
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
    print(f"Initialization result: {init_result}")
      # Get current time in IST using pytz directly
    import pytz
    import datetime
    ist_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    print(f"Current IST time: {ist_time}")
    
    # Run the strategy with force analysis to take a trade immediately
    print("Running strategy with force_analysis=True to take a trade...")
    run_result = strategy.run_strategy(force_analysis=True)
    print(f"Strategy run result: {run_result}")
    
    print("\nStrategy run completed!")
    
except Exception as e:
    import traceback
    print(f"Error running strategy: {e}")
    print(traceback.format_exc())

print("=" * 80)
