import csv
from datetime import datetime

class TradeMonitor:
    def __init__(self, symbol, direction, entry_price, quantity, stop_loss, target, max_holding_minutes=30, log_file='trade_log.csv'):
        self.symbol = symbol
        self.direction = direction.upper()
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.target = target
        self.max_holding_minutes = max_holding_minutes
        self.log_file = log_file

        self.entry_time = datetime.now()
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None

        self.max_up_pnl = 0.0
        self.max_down_pnl = 0.0
        self.max_up_pct = 0.0
        self.max_down_pct = 0.0

        self.initial_investment = entry_price * quantity
        self.closed = False

        self.log_event(f"TRADE ENTRY: {self.symbol} {self.direction} {self.entry_price} Qty={self.quantity} SL={self.stop_loss} Target={self.target}")

    def on_price_update(self, current_price):
        if self.closed:
            return

        pnl = (current_price - self.entry_price) * self.quantity if self.direction == 'BUY' else (self.entry_price - current_price) * self.quantity
        pnl_pct = (pnl / self.initial_investment) * 100 if self.initial_investment else 0.0

        # Update max up/down
        if pnl > self.max_up_pnl or self.max_up_pnl == 0.0:
            self.max_up_pnl = pnl
            self.max_up_pct = pnl_pct
        if pnl < self.max_down_pnl or self.max_down_pnl == 0.0:
            self.max_down_pnl = pnl
            self.max_down_pct = pnl_pct

        # Exit checks
        if (self.direction == 'BUY' and current_price <= self.stop_loss) or (self.direction == 'SELL' and current_price >= self.stop_loss):
            self.close_trade(current_price, 'STOPLOSS')
        elif (self.direction == 'BUY' and current_price >= self.target) or (self.direction == 'SELL' and current_price <= self.target):
            self.close_trade(current_price, 'TARGET')
        elif (datetime.now() - self.entry_time).total_seconds() > self.max_holding_minutes * 60:
            self.close_trade(current_price, 'MAX_DURATION')

    def close_trade(self, exit_price, reason):
        if self.closed:
            return
        self.exit_time = datetime.now()
        self.exit_price = exit_price
        self.exit_reason = reason
        self.closed = True
        self.log_event(f"TRADE EXIT: {self.symbol} {self.direction} Exit={exit_price} Reason={reason}")
        self.log_summary()

    def log_event(self, message):
        print(f"{datetime.now()} - {message}")

    def log_summary(self):
        # Write summary to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    'Entry DateTime', 'Symbol', 'Direction', 'Entry Price', 'Exit DateTime', 'Exit Price',
                    'Stop Loss', 'Target', 'Quantity', 'Gross P&L', 'Margin Required', '% Gain/Loss',
                    'Max Up (₹)', 'Max Down (₹)', 'Max Up (%)', 'Max Down (%)', 'Exit Reason'
                ])
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity if self.direction == 'BUY' else (self.entry_price - self.exit_price) * self.quantity
            margin_required = self.entry_price * self.quantity
            pct_gain_loss = (gross_pnl / margin_required) * 100 if margin_required else 0.0
            writer.writerow([
                self.entry_time, self.symbol, self.direction, self.entry_price, self.exit_time, self.exit_price,
                self.stop_loss, self.target, self.quantity, round(gross_pnl, 2), margin_required, round(pct_gain_loss, 2),
                round(self.max_up_pnl, 2), round(self.max_down_pnl, 2), round(self.max_up_pct, 2), round(self.max_down_pct, 2), self.exit_reason
            ])
