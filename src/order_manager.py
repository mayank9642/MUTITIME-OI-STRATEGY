import logging
import time
import uuid
from enum import Enum
import threading

class GTTOrderStatus(Enum):
    CANCELLED = 1
    TRIGGERED = 2
    PENDING = 3
    EXPIRED = 4
    ERROR = 5

class OrderManager:
    def __init__(self, broker_api=None, paper_trading=True, order_expiry_seconds=86400):
        self.broker_api = broker_api
        self.paper_trading = paper_trading
        self.order_expiry_seconds = order_expiry_seconds
        self._lock = threading.Lock()
        self.orders = {}  # order_id: order dict
        self.gtt_groups = {}  # group_id: set(order_id)
        logging.info("OrderManager initialized. Paper trading: %s", self.paper_trading)

    def place_gtt_order(self, symbol, side, qty, trigger_price, price=None, product_type="INTRADAY", tag="", group_id=None):
        """
        Place a GTT (Good Till Trigger) order that remains active until triggered
        """
        order_id = str(uuid.uuid4())
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "trigger_price": trigger_price,
            "price": price if price is not None else trigger_price,
            "productType": product_type,
            "status_code": GTTOrderStatus.PENDING.value,
            "tag": tag,
            "created_at": time.time(),
            "group_id": group_id,
            "error": None
        }
        with self._lock:
            self.orders[order_id] = order
            if group_id:
                if group_id not in self.gtt_groups:
                    self.gtt_groups[group_id] = set()
                self.gtt_groups[group_id].add(order_id)
        logging.info(f"[ORDER_MANAGER] GTT order placed: {order}")
        if self.paper_trading:
            return {"order_id": order_id, "status": "pending", "order": order}
        # TODO: Integrate with broker API for live trading
        # try:
        #     response = self.broker_api.place_gtt_order(order)
        #     ...
        # except Exception as e:
        #     order['status_code'] = GTTOrderStatus.ERROR.value
        #     order['error'] = str(e)
        #     logging.error(f"Broker API error placing GTT: {e}")
        #     return {"status": "error", "msg": str(e)}
        return {"status": "not_implemented", "msg": "Live GTT not implemented"}

    def check_gtt_order_status(self, order_id):
        """
        Check status of a specific GTT order
        """
        with self._lock:
            order = self.orders.get(order_id)
            if order:
                return order
            return {"status_code": 0, "msg": "Order not found"}

    def cancel_gtt_order(self, order_id, reason="User requested"): 
        """
        Cancel a specific GTT order
        """
        with self._lock:
            order = self.orders.get(order_id)
            if order and order['status_code'] == GTTOrderStatus.PENDING.value:
                order['status_code'] = GTTOrderStatus.CANCELLED.value
                order['cancelled_at'] = time.time()
                order['cancel_reason'] = reason
                logging.info(f"GTT order cancelled: {order}")
                # Remove from group if present
                group_id = order.get('group_id')
                if group_id and group_id in self.gtt_groups:
                    self.gtt_groups[group_id].discard(order_id)
                return {"status": "cancelled", "order_id": order_id}
            elif order:
                logging.warning(f"Cancel failed: Order not pending: {order}")
                return {"status": "not_pending", "order_id": order_id}
            else:
                logging.error(f"Cancel failed: Order not found: {order_id}")
                return {"status": "not_found", "order_id": order_id}

    def cancel_group_gtt_orders(self, group_id, except_order_id=None, reason="Mutual exclusivity"): 
        """
        Cancel all GTT orders in a group except the specified one
        """
        with self._lock:
            order_ids = self.gtt_groups.get(group_id, set()).copy()
            for oid in order_ids:
                if oid != except_order_id:
                    self.cancel_gtt_order(oid, reason=reason)

    def monitor_active_gtt_orders(self, get_price_func):
        """
        Monitor all active GTT orders and handle triggered orders
        get_price_func(symbol) should return the current price
        """
        now = time.time()
        triggered = []
        expired = []
        with self._lock:
            for order_id, order in list(self.orders.items()):
                if order['status_code'] != GTTOrderStatus.PENDING.value:
                    continue
                # Expiry check
                if now - order['created_at'] > self.order_expiry_seconds:
                    order['status_code'] = GTTOrderStatus.EXPIRED.value
                    order['expired_at'] = now
                    expired.append(order)
                    logging.info(f"GTT order expired: {order}")
                    continue
                symbol = order['symbol']
                trigger_price = order['trigger_price']
                side = order['side']
                try:
                    price = get_price_func(symbol)
                except Exception as e:
                    order['status_code'] = GTTOrderStatus.ERROR.value
                    order['error'] = str(e)
                    logging.error(f"Error getting price for {symbol}: {e}")
                    continue
                if price is None:
                    logging.warning(f"Skipping GTT trigger check for {symbol}: price is None")
                    continue
                if (side == 1 and price >= trigger_price) or (side == -1 and price <= trigger_price):
                    order['status_code'] = GTTOrderStatus.TRIGGERED.value
                    order['triggered_at'] = now
                    triggered.append(order)
                    logging.info(f"GTT order triggered: {order}")
                    # Mutual exclusivity: cancel others in group
                    group_id = order.get('group_id')
                    if group_id:
                        self.cancel_group_gtt_orders(group_id, except_order_id=order_id)
        return triggered

    def get_orders_by_status(self, status_code):
        """
        Return all orders with a given status code
        """
        with self._lock:
            return [o for o in self.orders.values() if o['status_code'] == status_code]

    def get_orders_by_symbol(self, symbol):
        with self._lock:
            return [o for o in self.orders.values() if o['symbol'] == symbol]

    def get_orders_by_tag(self, tag):
        with self._lock:
            return [o for o in self.orders.values() if o['tag'] == tag]

    def cleanup_expired_and_cancelled_orders(self):
        """
        Remove expired/cancelled/errored orders from memory (optional, for long-running sessions)
        """
        with self._lock:
            to_remove = [oid for oid, o in self.orders.items() if o['status_code'] in (
                GTTOrderStatus.CANCELLED.value, GTTOrderStatus.EXPIRED.value, GTTOrderStatus.ERROR.value)]
            for oid in to_remove:
                del self.orders[oid]
        logging.info(f"Cleaned up {len(to_remove)} expired/cancelled/error orders.")

    # --- Unit Test Helpers ---
    def _reset_all_orders(self):
        """
        For unit testing: reset all order state
        """
        with self._lock:
            self.orders.clear()
            self.gtt_groups.clear()
        logging.info("OrderManager state reset for unit testing.")

    # --- Broker API Integration Stubs ---
    def on_gtt_triggered(self, order_id):
        """
        Callback for broker API when a GTT is triggered (stub)
        """
        logging.info(f"Broker GTT triggered callback for order_id: {order_id}")
        # Implement integration as needed

    def on_gtt_cancelled(self, order_id):
        """
        Callback for broker API when a GTT is cancelled (stub)
        """
        logging.info(f"Broker GTT cancelled callback for order_id: {order_id}")
        # Implement integration as needed
