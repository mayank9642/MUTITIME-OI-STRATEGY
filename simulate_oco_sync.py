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

logging.info('Starting synchronous simulated OCO run (no callback)')

s = OpenInterestStrategy()
ce = 'NIFTY26MAR53900CE'
pe = 'NIFTY26MAR53600PE'
s.call_breakout_level = 1326.95
s.put_breakout_level = 1171.70
s.highest_call_oi_symbol = ce
s.highest_put_oi_symbol = pe
s.breakout_levels_fixed = True

# Ensure OrderManager will NOT invoke the callback (we will handle triggers synchronously)
try:
    s.order_manager.on_gtt_triggered = None
except Exception:
    pass

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
        # Ensure bracket_order_id is set synchronously; if not, place bracket directly to avoid race
        bracket_id = None
        for _ in range(5):
            bracket_id = s.active_trade.get('bracket_order_id')
            if bracket_id:
                break
            time.sleep(0.1)
        if not bracket_id:
            logging.warning('Bracket order id not set by execute_trade; placing bracket order directly to ensure determinism')
            try:
                cfg = s.config.get('strategy', {}) if s.config else {}
                sl_pct = float(cfg.get('stoploss_pct', 7))
                tgt_pct = float(cfg.get('target_pct', 7))
                stoploss = round(float(price) * (1 - sl_pct / 100.0), 2)
                target = round(float(price) * (1 + tgt_pct / 100.0), 2)
                bracket = s.order_manager.place_bracket_order(symbol=sym, side=1, qty=qty, entry_price=price, stoploss=stoploss, target=target, tag='BRACKET')
                logging.info(f"Direct bracket placement response: {bracket}")
                if isinstance(bracket, dict):
                    bid = bracket.get('order_id') or (bracket.get('order') or {}).get('order_id')
                    if bid:
                        s.active_trade['bracket_order_id'] = bid
                        logging.info(f"Assigned bracket_order_id to active_trade: {bid}")
            except Exception:
                logging.exception('Failed to place bracket directly')
        # cancel other group members
        try:
            s.order_manager.cancel_group_gtt_orders(group_id, except_order_id=order_id)
        except Exception:
            logging.exception('Error cancelling group orders')

# Give execute_trade a short moment to place bracket order and log details
time.sleep(0.2)

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
            # simulate processing of the filled bracket by the strategy
            try:
                logging.info(f"Processing filled bracket order: {f}")
                # mark exit in strategy
                s._close_active_trade(exit_reason='TARGET', exit_price=f.get('filled_price'))
            except Exception:
                logging.exception('Error processing filled bracket')
else:
    logging.error('No bracket order id found after execute_trade')

logging.info('Synchronous simulated OCO run (no callback) complete')
