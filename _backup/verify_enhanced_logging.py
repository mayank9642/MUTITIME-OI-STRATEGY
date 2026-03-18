#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.strategy import OpenInterestStrategy
    print("Enhanced logging system verification:")
    
    strategy = OpenInterestStrategy()
    print("✅ Strategy initialized successfully")
    
    methods = [
        'save_trade_history', 
        'load_current_balance', 
        'save_current_balance', 
        'update_balance_on_trade_completion', 
        'calculate_fyers_option_charges', 
        '_append_final_row_with_format'
    ]
    
    for method in methods:
        if hasattr(strategy, method):
            print(f"✅ {method} method exists")
        else:
            print(f"❌ {method} method missing")
    
    # Verify balance tracking variables
    balance_vars = ['current_balance', 'initial_balance']
    for var in balance_vars:
        if hasattr(strategy, var):
            print(f"✅ {var} variable exists")
        else:
            print(f"❌ {var} variable missing")
    
    print("✅ Enhanced logging integration verification complete!")
    
except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()