"""Quick backtest/sweep harness for SL/TG mapping (VIX + premium buckets).

Run from project root:
    python scripts/backtest_rr.py

This will load the project's config (if available), create a strategy instance,
set up a simple VIX grid and premium grid, and print the computed SL/TG values
for inspection. Output is CSV-like to make it easy to redirect to a file.
"""
import sys
from src.strategy import OpenInterestStrategy
from src.config import load_config

def main():
    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        pass
    s = OpenInterestStrategy()
    # Ensure compute method uses our test vix values by bypassing live hooks
    print("vix,premium,mode,entry,sl,tg")
    vix_values = [8.0, 9.5, 11.0, 12.5, 14.0]
    premiums = [100.0, 300.0, 500.0, 750.0, 1200.0]
    for v in vix_values:
        # Place a single cached vix value for the strategy to pick up
        s._vix_cache = [(None, float(v))]
        s._vix_cache_loaded = True
        for p in premiums:
            sl, tg, mode = s.compute_sl_tg(p)
            print(f"{v},{p},{mode},{p},{sl},{tg}")

if __name__ == '__main__':
    main()
