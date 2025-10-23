"""
Simple execution of the strategy to take a trade
"""
import os
import sys
import logging
import datetime
import time

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure more robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_run.log')
    ]
)

# Write to a simple output file for debugging
with open("simple_run_output.txt", "w") as f:
    f.write(f"Script started at {datetime.datetime.now()}\n")

print("=" * 80)
print("RUNNING TRADING STRATEGY")
print("=" * 80)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Log environment information
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Files in src directory: {os.listdir('src')}")
    
    # Import the fixed strategy
    logging.info("Attempting to import FixedOpenInterestStrategy...")
    with open("simple_run_output.txt", "a") as f:
        f.write("Importing FixedOpenInterestStrategy...\n")
    
    from src.fixed_strategy_updated import FixedOpenInterestStrategy
    
    logging.info("Import successful")
    with open("simple_run_output.txt", "a") as f:
        f.write("Import successful\n")
    
    # Create an instance
    logging.info("Creating strategy instance...")
    strategy = FixedOpenInterestStrategy()
    logging.info(f"Strategy created, type: {type(strategy).__name__}")
    
    # Initialize
    logging.info("Initializing strategy...")
    init_result = strategy.initialize_day()
    logging.info(f"Initialization result: {init_result}")
    
    # Run with force analysis to take a trade
    logging.info("Running strategy with force_analysis=True...")
    result = strategy.run_strategy(force_analysis=True)
    
    logging.info(f"Strategy execution result: {result}")
    
    # Check if a trade was taken
    if hasattr(strategy, 'active_trade') and strategy.active_trade:
        logging.info(f"Trade taken: {strategy.active_trade}")
    else:
        logging.info("No trade was taken")
        
    logging.info("Strategy execution completed!")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())

print("=" * 80)
