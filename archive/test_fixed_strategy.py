"""
Test script to verify the FixedOpenInterestStrategy class
"""
import os
import sys

# Ensure we can import from the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("Python path:", sys.path)

try:
    print("Trying to import FixedOpenInterestStrategy...")
    from src.fixed_strategy import FixedOpenInterestStrategy
    print("Successfully imported FixedOpenInterestStrategy")
    
    print("Creating instance of FixedOpenInterestStrategy...")
    strategy = FixedOpenInterestStrategy()
    print(f"Strategy instance type: {type(strategy).__name__}")
    
    print("Checking for initialize_day method...")
    if hasattr(strategy, 'initialize_day'):
        print("initialize_day method exists")
    else:
        print("ERROR: initialize_day method does not exist")
        
    print("Checking for update_trailing_stoploss method...")
    if hasattr(strategy, 'update_trailing_stoploss'):
        print("update_trailing_stoploss method exists")
    else:
        print("ERROR: update_trailing_stoploss method does not exist")
        
    print("Checking for run_strategy method...")
    if hasattr(strategy, 'run_strategy'):
        print("run_strategy method exists")
    else:
        print("ERROR: run_strategy method does not exist")
    
    print("Test completed successfully")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Module import path might be incorrect")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())
