import time
# Import run_strategy to configure logging as the main script does
import importlib.util
import importlib.machinery
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
run_strategy_path = os.path.join(root, 'run_strategy.py')
spec = importlib.util.spec_from_file_location('run_strategy', run_strategy_path)
run_strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_strategy)
from src.strategy import OpenInterestStrategy

s = OpenInterestStrategy()
# Simulate a WS tick for a non-canonical symbol
ticks = {'ltp': 203.95, 'symbol': 'NIFTY17MAR26C23500'}
# Call ws_price_update which should use the symbol_formatter and create a WS_UPDATE log (rate-limited)
s.ws_price_update('NIFTY17MAR26C23500', 'tick', ticks, ticks)

# Simulate active trade and PAPER STATUS logs
s.active_trade = {
    'symbol': 'NSE:NIFTY17MAR26C23500',
    'entry_price': 180.0,
    'stoploss': 170.0,
    'target': 240.0,
    'quantity': 1,
    'entry_time': time.time()
}
# Provide get_active_trade_ltp used by log_trade_update
s.get_active_trade_ltp = lambda: 203.95

# Log twice to test rate-limiting (PAPER_STATUS min interval is 5s; second should be skipped)
s.log_trade_update()
# Wait less than paper_status_min_seconds to test skip
time.sleep(2)
s.log_trade_update()

print('Validation script completed')
