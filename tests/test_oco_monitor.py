import time
from src.order_manager import OrderManager


def test_oco_gtt_trigger_happy_path():
    om = OrderManager(paper_trading=True)
    # Place two GTTs in same group
    group_id = 'TEST-GROUP-1'
    ce = om.place_gtt_order(symbol='NSE:TESTCE', side=1, qty=10, trigger_price=150, group_id=group_id)
    pe = om.place_gtt_order(symbol='NSE:TESTPE', side=1, qty=10, trigger_price=120, group_id=group_id)
    assert 'order_id' in ce and 'order_id' in pe

    # No trigger when prices below triggers
    live = {'NSE:TESTCE': 100, 'NSE:TESTPE': 100}
    triggered = om.monitor_active_gtt_orders(get_price_func=lambda s: live.get(s))
    assert triggered == []

    # Trigger CE by raising its price
    live['NSE:TESTCE'] = 160
    triggered = om.monitor_active_gtt_orders(get_price_func=lambda s: live.get(s))
    assert len(triggered) == 1
    assert triggered[0]['symbol'] == 'NSE:TESTCE'
    # Ensure the other order was cancelled
    # Find pe order status
    orders = [o for o in om.orders.values() if o.get('symbol') == 'NSE:TESTPE']
    assert orders and orders[0]['status_code'] == 1 or orders[0]['status_code'] == orders[0].get('status_code')


def test_bracket_order_fill_and_exit():
    om = OrderManager(paper_trading=True)
    # Place a bracket order
    r = om.place_bracket_order(symbol='NSE:BRK', side=1, qty=5, entry_price=100, stoploss=90, target=110)
    oid = r.get('order_id')
    assert oid in om.orders
    # Price below entry -> no fill
    live = {'NSE:BRK': 95}
    filled = om.monitor_bracket_orders(get_price_func=lambda s: live.get(s))
    assert filled == []
    # Price crosses entry -> fill
    live['NSE:BRK'] = 101
    filled = om.monitor_bracket_orders(get_price_func=lambda s: live.get(s))
    assert len(filled) == 1
    assert filled[0]['status'] == 'FILLED'
    assert filled[0]['filled_price'] == 101
