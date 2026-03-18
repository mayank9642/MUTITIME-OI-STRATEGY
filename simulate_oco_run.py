import logging
import logging.handlers
import time
import os
from src.strategy import OpenInterestStrategy

# Configure logging like run_strategy.py
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
fh = logging.handlers.RotatingFileHandler('logs/strategy.log', maxBytes=10*1024*1024, backupCount=5, delay=True)
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.info('Starting simulated OCO run')

s = OpenInterestStrategy()
# Set test symbols (use canonical-like strings; get_canonical_symbol will normalize)
ce = 'NIFTY26MAR53900CE'
pe = 'NIFTY26MAR53600PE'
# Set breakout levels
s.call_breakout_level = 1326.95
s.put_breakout_level = 1171.70
s.highest_call_oi_symbol = ce
s.highest_put_oi_symbol = pe
s.breakout_levels_fixed = True

# Place OCO GTT orders
s.place_oco_gtt_orders(ce_symbol=ce, ce_trigger=s.call_breakout_level, pe_symbol=pe, pe_trigger=s.put_breakout_level, qty=35)

# Now simulate ticks: start below triggers and then push CE over trigger
canon_ce = s.get_canonical_symbol(ce)
canon_pe = s.get_canonical_symbol(pe)
logging.info(f"Sim test canonical symbols: CE={canon_ce}, PE={canon_pe}")

# Initialize live prices below trigger
s.live_prices[canon_ce] = 1200.0
s.live_prices[canon_pe] = 900.0

# Simulate gradual price changes; after some seconds raise CE above trigger
for i in range(10):
    time.sleep(1)
    s.live_prices[canon_ce] += 15  # bump CE
    s.live_prices[canon_pe] += 5   # bump PE
    logging.info(f"[SIM TICK] {canon_ce}={s.live_prices[canon_ce]:.2f} | {canon_pe}={s.live_prices[canon_pe]:.2f}")

logging.info('Simulation complete; waiting a few seconds for monitor threads to process')
# Wait a bit to let background monitor threads detect fills and log
time.sleep(5)
logging.info('Sim run finished')
