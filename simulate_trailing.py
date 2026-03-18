import logging
import logging.handlers
import os
import time
from src.strategy import OpenInterestStrategy

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

logging.info('Starting trailing simulation')

s = OpenInterestStrategy()
raw = 'NIFTY26MAR53900CE'
canon = s.get_canonical_symbol(raw)
entry = 100.0
qty = 1
# Create active_trade manually (simulate that trade was entered)
s.active_trade = {
    'symbol': canon,
    'entry_price': entry,
    'entry_time': s.get_ist_datetime(),
    'quantity': qty,
    'side': 'BUY',
    'stoploss': round(entry * (1 - 0.20), 2),
    'target': round(entry * (1 + 0.40), 2),
    'trailing_stop_pct': float(s.config.get('strategy', {}).get('trailing_stop_pct', 8))
}
# Set initial live price
s.live_prices[canon] = entry
logging.info(f"Active trade initialized: {s.active_trade}")
# Sequence of prices to simulate
prices = [113.1, 120.1, 125.1, 130.1, 135.1]
for p in prices:
    s.live_prices[canon] = p
    logging.info(f"--- Simulating price update: {p}")
    s.log_trade_update()
    time.sleep(0.1)

logging.info('Trailing simulation complete')
