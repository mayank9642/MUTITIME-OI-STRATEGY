import sys
import os
import pandas as pd


def make_chain():
    # Build a synthetic option chain DataFrame covering CE and PE with two expiries
    rows = []
    expiry1 = '2026-04-02'
    expiry2 = '2026-04-09'

    # CE side: top-OI has low premium (40), second-highest has premium (60)
    rows.append({'option_type': 'CE', 'strike': 100, 'symbol': 'NSE:TEST0104CE', 'oi': 1000, 'change_oi': 0, 'ltp': 40.0, 'expiry': expiry1})
    rows.append({'option_type': 'CE', 'strike': 105, 'symbol': 'NSE:TEST0104CE', 'oi': 900, 'change_oi': 0, 'ltp': 60.0, 'expiry': expiry1})
    # PE side: top-OI satisfies premium
    rows.append({'option_type': 'PE', 'strike': 95, 'symbol': 'NSE:TEST0104PE', 'oi': 1200, 'change_oi': 0, 'ltp': 70.0, 'expiry': expiry1})

    # Add some farther expiry contracts to ensure expiry ordering works
    rows.append({'option_type': 'CE', 'strike': 110, 'symbol': 'NSE:TEST0904CE', 'oi': 2000, 'change_oi': 0, 'ltp': 30.0, 'expiry': expiry2})
    rows.append({'option_type': 'PE', 'strike': 90, 'symbol': 'NSE:TEST0904PE', 'oi': 800, 'change_oi': 0, 'ltp': 55.0, 'expiry': expiry2})

    return pd.DataFrame(rows)


def test_select_second_rank_if_top_below_threshold(monkeypatch):
    """Test that identify/run selection picks the second-ranked CE when the top-ranked CE's premium < min_premium."""
    # Import strategy lazily after monkeypatching the fetcher
    chain_df = make_chain()

    # Monkeypatch the fetcher to return our synthetic chain
    import src.fetch_option_oi_fyers as fetch_mod

    monkeypatch.setattr(fetch_mod, 'fetch_option_oi_fyers', lambda fyers, symbol, strikecount=50: chain_df)

    from src.strategy import OpenInterestStrategy

    s = OpenInterestStrategy()
    # Force the strategy to use a low min_premium so PE selection still works; set CE threshold to 50
    s.min_premium_threshold = 50.0
    s.max_strike_distance = 1000
    s.paper_trading = True

    res = s.run_oi_selection_and_place()

    assert res.get('ok') is True
    selected = res.get('selected')
    assert selected is not None
    # CE should pick the 105 (ltp 60) since top had ltp 40 < threshold
    ce = selected.get('CE')
    pe = selected.get('PE')
    assert ce is not None and int(ce['strike']) == 105
    assert pe is not None and int(pe['strike']) == 95
