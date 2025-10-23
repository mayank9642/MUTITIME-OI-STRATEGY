import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import unittest
import time
from src.order_manager import OrderManager, GTTOrderStatus

class DummyPriceProvider:
    def __init__(self, prices):
        self.prices = prices
    def __call__(self, symbol):
        return self.prices.get(symbol, 0)

class TestOrderManager(unittest.TestCase):
    def setUp(self):
        self.om = OrderManager(paper_trading=True, order_expiry_seconds=2)  # short expiry for test
        self.om._reset_all_orders()

    def test_place_and_lookup(self):
        resp = self.om.place_gtt_order('NIFTY23AUG18000CE', 1, 50, 100.0, tag='test')
        oid = resp['order_id']
        order = self.om.check_gtt_order_status(oid)
        self.assertEqual(order['status_code'], GTTOrderStatus.PENDING.value)
        self.assertEqual(order['symbol'], 'NIFTY23AUG18000CE')

    def test_trigger_order(self):
        self.om.place_gtt_order('NIFTY23AUG18000CE', 1, 50, 100.0, tag='trigger')
        prices = {'NIFTY23AUG18000CE': 101.0}
        triggered = self.om.monitor_active_gtt_orders(DummyPriceProvider(prices))
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]['status_code'], GTTOrderStatus.TRIGGERED.value)

    def test_cancel_order(self):
        resp = self.om.place_gtt_order('NIFTY23AUG18000CE', 1, 50, 100.0, tag='cancel')
        oid = resp['order_id']
        result = self.om.cancel_gtt_order(oid, reason='test cancel')
        self.assertEqual(result['status'], 'cancelled')
        order = self.om.check_gtt_order_status(oid)
        self.assertEqual(order['status_code'], GTTOrderStatus.CANCELLED.value)

    def test_group_mutual_exclusivity(self):
        g1 = self.om.place_gtt_order('NIFTY23AUG18000CE', 1, 50, 100.0, tag='group', group_id='G1')['order_id']
        g2 = self.om.place_gtt_order('NIFTY23AUG18000PE', -1, 50, 100.0, tag='group', group_id='G1')['order_id']
        prices = {'NIFTY23AUG18000CE': 101.0, 'NIFTY23AUG18000PE': 99.0}
        triggered = self.om.monitor_active_gtt_orders(DummyPriceProvider(prices))
        # Only one should be triggered, the other cancelled
        triggered_ids = [o['order_id'] for o in triggered]
        self.assertTrue(g1 in triggered_ids or g2 in triggered_ids)
        o1 = self.om.check_gtt_order_status(g1)
        o2 = self.om.check_gtt_order_status(g2)
        self.assertTrue(
            (o1['status_code'] == GTTOrderStatus.TRIGGERED.value and o2['status_code'] == GTTOrderStatus.CANCELLED.value) or
            (o2['status_code'] == GTTOrderStatus.TRIGGERED.value and o1['status_code'] == GTTOrderStatus.CANCELLED.value)
        )

    def test_expiry(self):
        resp = self.om.place_gtt_order('NIFTY23AUG18000CE', 1, 50, 100.0, tag='expiry')
        oid = resp['order_id']
        time.sleep(2.1)
        self.om.monitor_active_gtt_orders(lambda s: 0)  # price won't trigger
        order = self.om.check_gtt_order_status(oid)
        self.assertEqual(order['status_code'], GTTOrderStatus.EXPIRED.value)

    def test_cleanup(self):
        resp = self.om.place_gtt_order('NIFTY23AUG18000CE', 1, 50, 100.0, tag='cleanup')
        oid = resp['order_id']
        self.om.cancel_gtt_order(oid)
        self.om.cleanup_expired_and_cancelled_orders()
        order = self.om.orders.get(oid)
        self.assertIsNone(order)

if __name__ == '__main__':
    unittest.main()
