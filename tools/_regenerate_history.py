import sys, os
# Ensure project root is on sys.path so `from src...` works when invoked from arbitrary cwd
proj_root = r"c:\vs code projects\MUTITIME-OI-STRATEGY"
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from src.strategy import OpenInterestStrategy
import datetime, pytz

s = OpenInterestStrategy()
s.reset_balance(100000)

s.trade_history = [{
    "Entry DateTime": datetime.datetime.now(pytz.timezone("Asia/Kolkata")),
    "Index": "NIFTY",
    "Symbol": "NSE:NIFTY26MAR54200CE",
    "Direction": "BUY",
    "Entry Price": 1010.1,
    "Exit DateTime": datetime.datetime.now(pytz.timezone("Asia/Kolkata")),
    "Exit Price": 1093.25,
    "P&L": 2910.25,
    "Quantity": 35,
    "SL": 939.3929999999999,
    "Target": 1080.807,
    "trailing_stoploss": "",
    "Max Up": 2910.25,
    "Max Down": -2411.5,
    "Max Up %": 8.23,
    "Max Down %": -6.82
}]

s.save_trade_history()
print('regenerated trade_history files')
