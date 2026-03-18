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

logging.info('Starting synchronous simulated OCO run (direct)')

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

canon_ce = s.get_canonical_symbol(ce)
canon_pe = s.get_canonical_symbol(pe)

# Initialize live prices below triggers
s.live_prices[canon_ce] = 1200.0
s.live_prices[canon_pe] = 900.0

# Bump CE above trigger to simulate immediate trigger
s.live_prices[canon_ce] = s.call_breakout_level + 5

# Synchronously poll for triggered GTTs and handle them directly (no callback race)
triggered = s.order_manager.monitor_active_gtt_orders(get_price_func=lambda sym: s.live_prices.get(sym))
logging.info(f"Explicit monitor returned triggered: {triggered}")

if triggered:
    for order in triggered:
        group_id = order.get('group_id')
        order_id = order.get('order_id')
        sym = order.get('symbol')
        price = order.get('trigger_price') or order.get('price')
        qty = order.get('qty')
        logging.info(f"Converting triggered GTT to active trade (sync): {sym} @ {price}")
        s.execute_trade(symbol=sym, side='BUY', entry_price=price, quantity=qty)
        # cancel other group members
        try:
            s.order_manager.cancel_group_gtt_orders(group_id, except_order_id=order_id)
        except Exception:
            logging.exception('Error cancelling group orders')

# Now ensure bracket order id exists and simulate bracket fill
bracket_id = s.active_trade.get('bracket_order_id')
if bracket_id:
    logging.info(f"Bracket id present: {bracket_id}, simulating price to trigger bracket fill")
    # set live price to entry_price + small amount to trigger BUY fill
    s.live_prices[s.active_trade['symbol']] = s.active_trade['entry_price'] + 10
    time.sleep(0.1)
    filled = s.order_manager.monitor_bracket_orders(get_price_func=lambda sym: s.live_prices.get(sym))
    logging.info(f"Bracket monitor filled: {filled}")
    if filled:
        for f in filled:
            s.execute_trade(symbol=f.get('symbol'), side='BUY', entry_price=f.get('filled_price'), quantity=f.get('qty'))
else:
    logging.error('No bracket order id found after execute_trade')

logging.info('Synchronous simulated OCO run (direct) complete')
