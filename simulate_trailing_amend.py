"""
Simple simulation to exercise amend + cancel+replace behavior in paper-mode
"""
import time
from datetime import datetime
from src.strategy import OpenInterestStrategy

s = OpenInterestStrategy()
# Ensure paper trading
s.paper_trading = True
s.order_manager.paper_trading = True
# Load config defaults from existing config if available
s.config = {
    'strategy': {
        'use_trailing_stop': True,
        'trailing': {
            'prefer_amend': True,
            'min_move_pct': 0.5,
            'cooldown_secs': 1,
            'max_modifications': 10
        }
    }
}

symbol = 'NSE:NIFTY_TEST_01CE'
entry_price = 100.0
initial_sl = 80.0
target = 140.0
qty = 25

# Place initial bracket order
resp = s.order_manager.place_bracket_order(symbol=symbol, side=1, qty=qty, entry_price=entry_price, stoploss=initial_sl, target=target)
order_id = resp.get('order', {}).get('order_id')
print(f"Placed initial bracket order: {order_id}")

# Populate active_trade
s.active_trade = {
    'symbol': symbol,
    'entry_price': entry_price,
    'quantity': qty,
    'side': 'BUY',
    'stoploss': initial_sl,
    'target': target,
    'bracket_order_id': order_id,
    'entry_time': datetime.now()
}

# Simulate price steps that should trigger trailing updates
price_steps = [100.0, 112.0, 114.0, 126.0, 131.0, 150.0]
for p in price_steps:
    print('\n--- New price:', p)
    s._apply_trailing_stop(current_price=p, entry_price=entry_price)
    # Inspect the simulated order record
    with s.order_manager._lock:
        order = s.order_manager.orders.get(s.active_trade.get('bracket_order_id'))
        print('Active trade stoploss (in-memory):', s.active_trade.get('stoploss'))
        print('Order record stoploss:', order.get('stoploss') if order else 'ORDER NOT FOUND')
    time.sleep(0.2)

print('\nSimulation complete.')
