import logging
import logging.handlers
import time
import os
from src.strategy import OpenInterestStrategy

# Configure logging to console and file
os.makedirs('logs', exist_ok=True)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    try:
        h.close()
    except Exception:
        pass
    logger.removeHandler(h)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.handlers.RotatingFileHandler('logs/strategy.log', maxBytes=5*1024*1024, backupCount=3, delay=True)
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.info('Starting direct execute simulation (no GTT/OCO)')

s = OpenInterestStrategy()
# Prepare a canonical symbol and entry price
raw_sym = 'NIFTY26MAR53900CE'
canon = s.get_canonical_symbol(raw_sym)
entry_price = 1200.5
qty = 10
# Ensure live price map contains the contract
s.live_prices[canon] = entry_price

# Directly call execute_trade which should place a bracket order and start PAPER STATUS
logging.info(f"Calling execute_trade directly for {canon} @ {entry_price} qty={qty}")
ok = s.execute_trade(symbol=canon, side='BUY', entry_price=entry_price, quantity=qty)
logging.info(f"execute_trade returned: {ok}")
# Brief pause to let execute_trade place bracket
time.sleep(0.2)

# Inspect active trade and bracket id
bracket_id = s.active_trade.get('bracket_order_id')
logging.info(f"Active trade: {s.active_trade}")
if not bracket_id:
    logging.error('No bracket_order_id found after execute_trade')
else:
    logging.info(f'Bracket order id: {bracket_id} found — simulating price move to fill')
    # Simulate price move above entry to fill the bracket (BUY fills when price >= entry_price)
    s.live_prices[canon] = entry_price + 5.0
    # Call monitor_bracket_orders synchronously to pick up fills
    filled = s.order_manager.monitor_bracket_orders(get_price_func=lambda sym: s.live_prices.get(sym))
    logging.info(f"monitor_bracket_orders returned: {filled}")
    if filled:
        for f in filled:
            logging.info(f"Simulated filled order: {f}")
            # Close trade via internal close logic
            s._close_active_trade(exit_reason='SIM_TARGET', exit_price=f.get('filled_price'))

logging.info('Direct execute simulation complete')
