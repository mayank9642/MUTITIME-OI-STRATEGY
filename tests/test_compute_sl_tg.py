import pytest
from src.strategy import OpenInterestStrategy


def test_compute_sl_tg_vix_less_10_low_premium():
    s = OpenInterestStrategy()
    s.live_prices['NSE:VIX'] = 9.0
    stop, tgt, mode = s._compute_sl_tg(400)
    # For vix<10 and entry<=500, target pct should be ~7% -> target = 400*1.07 = 428.0
    assert pytest.approx(tgt, rel=1e-3) == 400 * 1.07
    assert mode.startswith('banknifty_vix_based')


def test_compute_sl_tg_vix_between_10_12_high_premium():
    s = OpenInterestStrategy()
    s.live_prices['NSE:VIX'] = 11.0
    stop, tgt, mode = s._compute_sl_tg(600)
    # For 10<=vix<12 and entry>500, target pct should be 5% -> target = 600*1.05
    assert pytest.approx(tgt, rel=1e-3) == 600 * 1.05
    assert mode.startswith('banknifty_vix_based')


def test_compute_sl_tg_vix_ge_12_low_premium():
    s = OpenInterestStrategy()
    s.live_prices['NSE:VIX'] = 13.0
    stop, tgt, mode = s._compute_sl_tg(300)
    # For vix>=12 and entry<=500, target pct should be 12% -> target = 300*1.12
    assert pytest.approx(tgt, rel=1e-3) == 300 * 1.12
    assert mode.startswith('banknifty_vix_based')
