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
        # For throttling repetitive price-missing warnings per-symbol
        self._last_price_warn = {}
        self._price_warn_cooldown = 30.0
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

    def modify_order_stoploss(self, order_id, new_stoploss):
        """
        Modify the stoploss of an existing bracket/GTT order in paper mode.
        Returns a dict mimicking broker response: {'s':'ok','code':1102,'id':order_id, 'order': order}
        """
        # If running live and broker_api is available, try to call broker modify gtt API
        if not self.paper_trading and self.broker_api:
            # Try broker modify with a small retry loop for transient failures
            retry_count = 2
            retry_delay = 0.5
            try:
                data = {
                    "id": order_id,
                    "orderInfo": {
                        "leg1": {
                            "triggerPrice": float(new_stoploss),
                            "price": float(new_stoploss)
                        }
                    }
                }
                for attempt in range(1, retry_count + 1):
                    # The fyers client exposes modify_gtt_order in its model
                    try:
                        if hasattr(self.broker_api, 'modify_gtt_order'):
                            resp = self.broker_api.modify_gtt_order(data=data)
                            logging.info(f"[ORDER_MANAGER][BROKER] modify_gtt_order response (attempt {attempt}): {resp}")
                            # If success-ish, return immediately
                            if isinstance(resp, dict) and resp.get('s') == 'ok':
                                return resp
                            # If API returns non-ok, log and decide whether to retry
                            logging.warning(f"[ORDER_MANAGER][BROKER] modify_gtt_order non-ok response on attempt {attempt}: {resp}")
                        elif hasattr(self.broker_api, 'modify_order'):
                            resp = self.broker_api.modify_order(data={"id": order_id, "stopPrice": float(new_stoploss)})
                            logging.info(f"[ORDER_MANAGER][BROKER] modify_order response (attempt {attempt}): {resp}")
                            if isinstance(resp, dict) and resp.get('s') == 'ok':
                                return resp
                            logging.warning(f"[ORDER_MANAGER][BROKER] modify_order non-ok response on attempt {attempt}: {resp}")
                        else:
                            logging.debug("[ORDER_MANAGER][BROKER] No modify API found on broker client; will fall back to in-memory")
                            break
                    except Exception as e:
                        logging.exception(f"[ORDER_MANAGER][BROKER] Exception during modify attempt {attempt}: {e}")
                    # small backoff before retrying
                    if attempt < retry_count:
                        time.sleep(retry_delay * attempt)
            except Exception as e:
                logging.exception(f"[ORDER_MANAGER][BROKER] Exception preparing modify request: {e}")

        # Otherwise operate on the simulated in-memory order store
        with self._lock:
            order = self.orders.get(order_id)
            if not order:
                logging.error(f"Modify failed: Order not found: {order_id}")
                return {"s": "error", "code": 404, "msg": "order_not_found", "id": order_id}
            # Only allow modification of pending orders
            status = order.get('status') or order.get('status_code')
            if status not in ("PENDING", GTTOrderStatus.PENDING.value):
                logging.warning(f"Modify failed: Order not pending: {order}")
                return {"s": "error", "code": 400, "msg": "order_not_pending", "id": order_id}

            # If it's a bracket order, update the stoploss field
            if order.get('type') == 'BRACKET':
                old = order.get('stoploss')
                order['stoploss'] = new_stoploss
                logging.info(f"[ORDER_MANAGER] Bracket order amended stoploss: {order_id} {old} -> {new_stoploss}")
                return {"s": "ok", "code": 1102, "id": order_id, "order": order}

            # If it's a GTT order with orderInfo structure, try to update leg prices
            if 'trigger_price' in order or 'triggerPrice' in order:
                # generic update
                if 'stoploss' in order:
                    order['stoploss'] = new_stoploss
                logging.info(f"[ORDER_MANAGER] Order amended stoploss-like field for: {order_id}")
                return {"s": "ok", "code": 1102, "id": order_id, "order": order}

            logging.warning(f"Modify: Unsupported order type for modify: {order}")
            return {"s": "error", "code": 400, "msg": "unsupported_order_type", "id": order_id}

    def cancel_and_replace_stop(self, order_id, new_stoploss):
        """
        Cancel the given bracket order and place a replacement bracket with updated stoploss.
        Returns the new order response dict from place_bracket_order or an error.
        """
        with self._lock:
            order = self.orders.get(order_id)
            if not order:
                logging.error(f"Cancel+Replace failed: Order not found: {order_id}")
                return {"s": "error", "code": 404, "msg": "order_not_found", "id": order_id}
            # Capture necessary fields to recreate
            symbol = order.get('symbol')
            side = order.get('side')
            qty = order.get('qty') or order.get('quantity') or order.get('qty', None)
            entry_price = order.get('entry_price')
            target = order.get('target')
        # Cancel outside lock (cancel_order will re-acquire lock)
        cancel_resp = self.cancel_order(order_id, reason="cancel+replace by strategy")
        if cancel_resp.get('status') not in ("cancelled",):
            logging.error(f"Cancel+Replace: failed to cancel original order: {cancel_resp}")
            return {"s": "error", "code": 400, "msg": "cancel_failed", "id": order_id}
        # Place new bracket
        try:
            resp = self.place_bracket_order(symbol=symbol, side=side, qty=qty, entry_price=entry_price, stoploss=new_stoploss, target=target)
            logging.info(f"[ORDER_MANAGER] Cancel+Replace placed new bracket: old={order_id} new={resp.get('order', {}).get('order_id')}")
            return resp
        except Exception as e:
            logging.exception(f"Cancel+Replace: Exception placing replacement order: {e}")
            return {"s": "error", "code": 500, "msg": str(e), "id": order_id}

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
                    # Throttle repeated warnings so logs don't flood when websocket hasn't populated prices yet
                    nowt = time.time()
                    last_warn = self._last_price_warn.get(symbol, 0)
                    if nowt - last_warn >= self._price_warn_cooldown:
                        logging.warning(f"Skipping GTT trigger check for {symbol}: price is None (will retry)")
                        self._last_price_warn[symbol] = nowt
                    else:
                        logging.debug(f"Skipping GTT trigger check for {symbol}: price missing (throttled)")
                    continue
                if (side == 1 and price >= trigger_price) or (side == -1 and price <= trigger_price):
                    order['status_code'] = GTTOrderStatus.TRIGGERED.value
                    order['triggered_at'] = now
                    triggered.append(order)
                    logging.info(f"GTT order triggered: {order}")
                    # Notify any registered callback about the triggered order so higher-level
                    # components (strategy) can react immediately.
                    try:
                        if hasattr(self, 'on_gtt_triggered') and callable(self.on_gtt_triggered):
                            # Callbacks should handle exceptions; run in a separate thread to avoid blocking
                            try:
                                threading.Thread(target=lambda o=order: self.on_gtt_triggered(o), daemon=True).start()
                            except Exception:
                                # best-effort; if thread spawn fails, call directly (caught below)
                                self.on_gtt_triggered(order)
                    except Exception as e:
                        logging.exception(f"Error invoking on_gtt_triggered callback: {e}")
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

    def get_orders_by_group(self, group_id):
        """Return all orders belonging to a group id (GTT OCO groups)"""
        with self._lock:
            ids = self.gtt_groups.get(group_id, set()).copy()
            return [self.orders.get(i) for i in ids if i in self.orders]

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

    def place_bracket_order(self, symbol, side, qty, entry_price, stoploss, target, tag="BRACKET"):
        """
        Place a simulated bracket order (entry + stoploss + target) for paper-trading.
        Returns order dict with order_id and status 'PENDING'.
        """
        order_id = str(uuid.uuid4())
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "stoploss": stoploss,
            "target": target,
            "status": "PENDING",
            "type": "BRACKET",
            "created_at": time.time(),
            "tag": tag,
        }
        with self._lock:
            self.orders[order_id] = order
        logging.info(f"[ORDER_MANAGER] Bracket order placed: {order}")
        if self.paper_trading:
            return {"order_id": order_id, "status": "pending", "order": order}
        return {"status": "not_implemented", "msg": "Live bracket orders not implemented"}

    def monitor_bracket_orders(self, get_price_func):
        """
        Monitor bracket orders for simulated fills.
        For BUY brackets: if market price >= entry_price -> mark FILLED.
        Returns list of filled order dicts.
        """
        now = time.time()
        filled = []
        with self._lock:
            for order_id, order in list(self.orders.items()):
                if order.get('type') != 'BRACKET' or order.get('status') != 'PENDING':
                    continue
                symbol = order.get('symbol')
                try:
                    price = get_price_func(symbol)
                except Exception as e:
                    order['error'] = str(e)
                    logging.error(f"Error getting price for {symbol} while monitoring bracket orders: {e}")
                    continue
                if price is None:
                    continue
                side = order.get('side')
                entry_price = order.get('entry_price')
                # BUY: fill when price >= entry_price; SELL: fill when price <= entry_price
                if (side == 1 and price >= entry_price) or (side == -1 and price <= entry_price):
                    order['status'] = 'FILLED'
                    order['filled_at'] = now
                    order['filled_price'] = price
                    filled.append(order)
                    logging.info(f"[ORDER_MANAGER] Bracket order FILLED: {order}")
        return filled

    def cancel_order(self, order_id, reason="User requested"):
        """
        Cancel a generic order (GTT or BRACKET).
        """
        with self._lock:
            order = self.orders.get(order_id)
            if not order:
                logging.error(f"Cancel failed: Order not found: {order_id}")
                return {"status": "not_found", "order_id": order_id}
            status = order.get('status') or order.get('status_code')
            if status in ("PENDING", GTTOrderStatus.PENDING.value):
                order['status'] = 'CANCELLED'
                order['cancelled_at'] = time.time()
                order['cancel_reason'] = reason
                logging.info(f"Order cancelled: {order}")
                return {"status": "cancelled", "order_id": order_id}
            else:
                logging.warning(f"Cancel failed: Order not pending: {order}")
                return {"status": "not_pending", "order_id": order_id}

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
