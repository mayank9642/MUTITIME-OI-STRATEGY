"""
Smoke test to verify PAPER STATUS logging and trailing SL behavior.
Runs a short simulation that creates an OpenInterestStrategy, injects an active trade,
updates live_prices and invokes the PAPER STATUS runner for a few seconds.
"""
import sys
import time
import logging
import threading
from datetime import datetime

# Ensure project src is importable
sys.path.append('c:/vs code projects/MUTITIME-OI-STRATEGY')

# Configure logging to console for immediate feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.strategy import OpenInterestStrategy

s = OpenInterestStrategy()

# Prepare a fake active trade and initial LTP so get_active_trade_ltp returns a value
sym = 'TEST:SYM'
now = datetime.now()
s.active_trade = {
    'symbol': sym,
    'entry_price': 100.0,
    'stoploss': 80.0,
    'target': 150.0,
    'quantity': 25,
    'entry_time': now,
}
# Ensure live_prices contains the contract LTP
s.live_prices[sym] = 100.0

# Immediate single-shot status
logging.info("[SMOKE] Emitting immediate PAPER STATUS snapshot")
s.log_trade_update()

# Start a short runner that simulates price moves and calls log_trade_update periodically

def runner():
    logging.info("[SMOKE] Paper status runner starting")
    for i in range(8):
        try:
            # Simulate upward movement
            s.live_prices[sym] = s.live_prices.get(sym, 100.0) + (i * 1.5)
            s.log_trade_update()
        except Exception:
            logging.exception("[SMOKE] Exception while running paper status")
        time.sleep(1)
    logging.info("[SMOKE] Paper status runner finished")

th = threading.Thread(target=runner, name='SmokePaperStatusRunner', daemon=True)
th.start()
# Wait for it to finish
th.join(timeout=12)
print('SMOKE TEST DONE')
