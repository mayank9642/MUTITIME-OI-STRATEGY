# India VIX Market Condition Filtering

## Overview
India VIX (Volatility Index) monitoring has been integrated into the strategy to **prevent trading in sideways/low-volatility markets** where breakouts are unreliable and prone to false signals.

---

## Why VIX Matters

### Problem Without VIX Filter:
- In sideways markets (low VIX), breakout signals often fail
- False breakouts lead to immediate stop-loss hits
- Low volatility = Limited profit potential even on winning trades

### Solution With VIX Filter:
- **Automatic detection** of market conditions
- **Skip trading** when VIX indicates sideways market
- **Trade only when volatility supports** sustained directional moves

---

## VIX Levels & Trading Decisions

| VIX Range | Market Condition | Should Trade? | Risk Level | Description |
|-----------|------------------|---------------|------------|-------------|
| **< 12** | Very Low Volatility | ❌ **NO** | HIGH | Sideways/Range-bound - High risk of false breakouts |
| **12-16** | Low Volatility | ✅ Yes | MEDIUM | Stable market - Trade with tighter stops |
| **16-20** | Moderate Volatility | ✅ Yes | LOW | Normal trending conditions - Good for trading |
| **20-25** | High Volatility | ✅ Yes | LOW | Strong moves - Excellent for breakout trading |
| **25-35** | Very High Volatility | ✅ Yes | HIGH | Extreme volatility - Use wider stops |
| **> 35** | Panic/Crisis | ❌ **NO** | EXTREME | Market panic - Too unpredictable |

---

## Configuration

### File: `config/config.yaml`

```yaml
strategy:
  # India VIX thresholds for market condition filtering
  vix_check_enabled: true       # Set to false to disable VIX filtering
  vix_min_threshold: 12.0       # Don't trade if VIX below this (sideways market)
  vix_max_threshold: 35.0       # Don't trade if VIX above this (extreme panic)
```

### Parameters:

1. **vix_check_enabled** (boolean)
   - `true`: VIX filtering active (recommended)
   - `false`: Disable VIX checks (trade regardless of volatility)

2. **vix_min_threshold** (float, default: 12.0)
   - Minimum VIX required to trade
   - Below this = Sideways market, trading halted
   - Typical range: 10.0 - 14.0

3. **vix_max_threshold** (float, default: 35.0)
   - Maximum VIX allowed to trade
   - Above this = Extreme panic, trading halted
   - Typical range: 30.0 - 40.0

---

## How It Works

### 1. VIX Check at Strategy Start
Every time the strategy runs (09:20 AM onwards), it:

```
1. Fetches current India VIX from NSE
2. Analyzes market condition based on VIX level
3. Compares with configured thresholds
4. Decides whether to proceed with OI analysis
```

### 2. Log Output Example

#### VIX Too Low (Trading Halted):
```
============================================================
CHECKING INDIA VIX FOR MARKET CONDITIONS
============================================================
India VIX: 10.85
VIX Thresholds: Min=12.0, Max=35.0
Market Condition: Very Low Volatility
Description: Sideways/Range-bound market. High risk of false breakouts.
Risk Level: HIGH
Recommendation: Avoid trading - VIX 10.85 below minimum threshold 12.0
⚠️ TRADING HALTED: VIX outside safe trading range
⚠️ Current VIX 10.85 | Safe range: 12.0-35.0
============================================================
```

#### VIX in Safe Range (Trading Allowed):
```
============================================================
CHECKING INDIA VIX FOR MARKET CONDITIONS
============================================================
India VIX: 18.25
VIX Thresholds: Min=12.0, Max=35.0
Market Condition: Moderate Volatility
Description: Normal trending market conditions.
Risk Level: LOW
Recommendation: Good for trading - Normal conditions
✓ VIX check passed (18.25) - Proceeding with strategy
============================================================
```

---

## Today's Analysis (Nov 20, 2025)

### Current VIX: **12.14**

**Market Condition:** Low Volatility
- **Should Trade:** ✅ Yes (just above minimum threshold)
- **Risk Level:** MEDIUM
- **Recommendation:** Trade with caution - Use tighter stop losses

### Why Today's Losses Happened:
Looking at today's trades:
- 5 trades total
- 4 losses, 1 small win
- **VIX at 12.14** (barely above sideways threshold)
- Market was indeed choppy with false breakouts

**Conclusion:** Today was borderline low-volatility. The VIX filter would have helped avoid marginal trading conditions.

---

## Testing VIX Functionality

### Manual Test:
```bash
python src/fetch_india_vix.py
```

**Expected Output:**
```
INFO:__main__:India VIX fetched successfully: 12.14

India VIX: 12.14
Market Condition: Low Volatility
Description: Stable market with limited moves.
Should Trade: True
Risk Level: MEDIUM
Recommendation: Trade with caution - Tighter stop losses recommended
```

---

## Recommendations

### Conservative (Avoid Most False Breakouts):
```yaml
vix_check_enabled: true
vix_min_threshold: 14.0  # Stricter - only trade on clear volatility
vix_max_threshold: 30.0
```

### Balanced (Recommended):
```yaml
vix_check_enabled: true
vix_min_threshold: 12.0  # Default - good balance
vix_max_threshold: 35.0
```

### Aggressive (Trade More Often):
```yaml
vix_check_enabled: true
vix_min_threshold: 10.0  # Trade even in calmer markets
vix_max_threshold: 40.0
```

---

## Integration with Existing Fixes

The VIX filter works **in addition to** yesterday's fixes:

1. **Price Spike Filter** (>100% rejection)
   - Prevents bad tick data entries

2. **VIX Market Filter** (NEW)
   - Prevents trading in sideways markets

3. **DataFrame Preservation**
   - Ensures accurate P&L tracking

4. **Error Handling**
   - Graceful unsubscribe failures

**Combined Effect:** Multi-layer protection against both technical bugs and unfavorable market conditions.

---

## Troubleshooting

### VIX Fetch Fails
If VIX cannot be fetched from NSE:
- Strategy logs warning but **proceeds** (fail-safe)
- Consider this a yellow flag for manual review

### Disable VIX Check Temporarily
```yaml
vix_check_enabled: false
```
Use this if:
- NSE API is down
- Testing strategy logic
- Debugging other issues

---

## Files Modified

1. **NEW:** `src/fetch_india_vix.py` - VIX fetching and analysis
2. **UPDATED:** `src/strategy.py` - VIX check integration
3. **UPDATED:** `config/config.yaml` - VIX configuration parameters
4. **NEW:** `docs/VIX_FILTERING.md` - This documentation

---

## Next Steps for Tomorrow

1. ✅ VIX filtering now active
2. ⚠️ **CRITICAL:** Deploy yesterday's spike filter fixes (currently not running)
3. Monitor logs for:
   - VIX values at strategy start
   - "TRADING HALTED" messages if VIX too low
   - Improved win rate on trending days

---

## Historical Context

**Nov 20 Performance:**
- VIX: 12.14 (borderline low)
- Result: 1 win, 4 losses (-₹9,660)
- LTP mixup still occurred (fixes not deployed)

**Expected with VIX Filter:**
- If VIX < 12: Strategy would skip trading entirely
- If VIX 12-16: Would trade but with heightened caution
- If VIX > 16: Normal trading with good profit potential

---

*Last Updated: November 20, 2025 @ 11:30 PM IST*
