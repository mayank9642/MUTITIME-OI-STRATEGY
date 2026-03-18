import logging
import logging.handlers
import time
import os
from src.strategy import OpenInterestStrategy

# Configure logging
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

logging.info('Starting simulated OCO run 2')

s = OpenInterestStrategy()
ce = 'NIFTY26MAR53900CE'
pe = 'NIFTY26MAR53600PE'
s.call_breakout_level = 1326.95
s.put_breakout_level = 1171.70
s.highest_call_oi_symbol = ce
s.highest_put_oi_symbol = pe
s.breakout_levels_fixed = True

# Place OCO GTT orders
s.place_oco_gtt_orders(ce_symbol=ce, ce_trigger=s.call_breakout_level, pe_symbol=pe, pe_trigger=s.put_breakout_level, qty=35)

# canonical keys
canon_ce = s.get_canonical_symbol(ce)
canon_pe = s.get_canonical_symbol(pe)

# Initialize live prices below triggers
s.live_prices[canon_ce] = 1200.0
s.live_prices[canon_pe] = 900.0

# Simulate ticks quickly (shortened for fast CI/demo runs), then explicitly check/order_manager for triggers
for i in range(4):
    time.sleep(0.05)
    s.live_prices[canon_ce] += 15
    s.live_prices[canon_pe] += 5
    logging.info(f"[SIM TICK] {canon_ce}={s.live_prices[canon_ce]:.2f} | {canon_pe}={s.live_prices[canon_pe]:.2f}")

# Explicitly poll order_manager for triggered GTT orders
# For demo: ensure CE price crosses trigger quickly so monitor finds the triggered GTT
s.live_prices[canon_ce] = s.call_breakout_level + 10
triggered = s.order_manager.monitor_active_gtt_orders(get_price_func=lambda sym: s.live_prices.get(sym))
logging.info(f"Explicit monitor returned triggered: {triggered}")

if triggered:
    for order in triggered:
        if order.get('group_id'):
            group_id = order['group_id']
        order_id = order.get('order_id')
        sym = order.get('symbol')
        price = order.get('trigger_price')
        qty = order.get('qty')
        logging.info(f"Converting triggered GTT to active trade: {sym} @ {price}")
        s.execute_trade(symbol=sym, side='BUY', entry_price=price, quantity=qty)
        # cancel other group members (monitor_active_gtt_orders already cancels, but be explicit)
        try:
            s.order_manager.cancel_group_gtt_orders(group_id, except_order_id=order_id)
        except Exception:
            logging.exception('Error cancelling group orders')

# Now simulate bracket fill: set price >= entry_price for bracket
bracket_id = s.active_trade.get('bracket_order_id')
if bracket_id:
    # Set live price to entry (fast fill for demo)
    s.live_prices[s.active_trade['symbol']] = s.active_trade['entry_price'] + 10
    time.sleep(0.05)
    filled = s.order_manager.monitor_bracket_orders(get_price_func=lambda sym: s.live_prices.get(sym))
    logging.info(f"Bracket monitor filled: {filled}")

logging.info('Simulated OCO run 2 complete')
