# DEPLOYMENT CHECKLIST - Nov 20, 2025 @ 11:30 PM

## ⚠️ CRITICAL: Yesterday's Fixes NOT Deployed Today!

Today's 10:31 AM trade **still had the 211.85 LTP mixup**, proving yesterday's fixes were never deployed.

---

## What Happened Today (Nov 20)

### Trades Summary:
- **Total:** 5 trades
- **Profitable:** 1 trade (+₹101.50)
- **Losses:** 4 trades (-₹9,762)
- **Net P&L:** -₹9,660.50

### Critical Issues:
1. ✅ **LTP Mixup OCCURRED AGAIN** (10:31 AM)
   - Entry at 211.85 when actual price was ~74
   - Loss: ₹4,816 (-64.95%)
   - **Root Cause:** Yesterday's fixes not running

2. 📊 **Market Conditions:**
   - India VIX: 12.14 (borderline low volatility)
   - 4/5 trades lost due to choppy sideways market

---

## ALL FIXES TO DEPLOY BEFORE TOMORROW

### 1. Price Spike Filter (CRITICAL - From Nov 19)
**File:** `src/strategy.py` (line ~360)

**What it does:**
- Rejects any single-tick price change > 100%
- Prevents false entries from bad Fyers data
- Would have prevented today's 211.85 entry

**Status:** ⚠️ Code written but NOT deployed/running

---

### 2. DataFrame Preservation (From Nov 19)
**File:** `src/strategy.py` (line ~1070)

**What it does:**
- Maintains ltp_df row during trade entry
- Creates row with entry price if missing
- Prevents "No LTP found" warnings

**Status:** ⚠️ Code written but NOT deployed/running

---

### 3. Graceful Unsubscribe (From Nov 19)
**File:** `src/strategy.py` (lines 640, 925)

**What it does:**
- Wraps unsubscribe calls in try-except
- Prevents keyword argument errors
- Non-critical but improves stability

**Status:** ⚠️ Code written but NOT deployed/running

---

### 4. India VIX Filtering (NEW - Nov 20)
**Files:** 
- `src/fetch_india_vix.py` (NEW)
- `src/strategy.py` (updated with VIX check)
- `config/config.yaml` (VIX thresholds added)

**What it does:**
- Fetches India VIX before strategy runs
- Halts trading if VIX < 12 (sideways market)
- Halts trading if VIX > 35 (extreme panic)
- Would have warned about today's borderline conditions

**Status:** ✅ Code complete, needs testing in production

---

## Pre-Flight Checklist for Tomorrow

### Step 1: Verify Files Changed
```bash
# Check that strategy.py has all fixes
python -c "import src.strategy; print('Strategy loaded successfully')"

# Check VIX functionality
python src/fetch_india_vix.py

# Verify config has VIX settings
python -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(f'VIX enabled: {c[\"strategy\"][\"vix_check_enabled\"]}')"
```

### Step 2: Test Syntax
```bash
# Compile check
python -m py_compile src/strategy.py
python -m py_compile src/fetch_india_vix.py

# Should print "No errors" if successful
```

### Step 3: Restart Scheduler
**IMPORTANT:** The scheduler might be caching old code!

```bash
# Kill any running scheduler processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *oi_scheduler*"

# OR manually stop the scheduler terminal

# Restart fresh
python src/oi_strategy_scheduler.py
```

---

## Expected Behavior Tomorrow (Nov 21)

### At 9:20 AM:

1. **VIX Check:**
```
============================================================
CHECKING INDIA VIX FOR MARKET CONDITIONS
============================================================
India VIX: [actual value]
VIX Thresholds: Min=12.0, Max=35.0
Market Condition: [condition]
✓ VIX check passed - Proceeding with strategy
============================================================
```

2. **If Spike Occurs:**
```
WARNING: SPIKE REJECTED: NSE:NIFTY25NOV26000PE | Previous: 74.50 | New: 211.85
```
(No trade entry - protection active!)

3. **Normal Trade Entry:**
```
INFO: Breakout detected for NSE:NIFTY25NOV26000PE at price 91.0
[ORDER_MANAGER] Paper trade order placed: {..., 'price': 91.0, ...}
TRADE_UPDATE | Entry: 91.0 | LTP: 91.0 | ...
```

---

## What to Monitor Tomorrow

### Success Indicators:
- ✅ VIX logs appear at strategy start
- ✅ If VIX < 12: "TRADING HALTED" message
- ✅ If bad tick occurs: "SPIKE REJECTED" message
- ✅ No "No LTP found in DataFrame" warnings
- ✅ No entries at absurd prices like 211.85

### Red Flags:
- ❌ No VIX logs (filter not running)
- ❌ No spike warnings despite volatility
- ❌ Entry at clearly wrong price
- ❌ "No LTP found" warnings continue

---

## File Summary

### Modified Files:
1. `src/strategy.py` - 3 fixes from Nov 19 + VIX integration
2. `config/config.yaml` - VIX thresholds added
3. `src/fetch_india_vix.py` - NEW file created

### Documentation Created:
1. `CRITICAL_FIXES_NOV20.md` - Yesterday's fixes
2. `docs/VIX_FILTERING.md` - VIX filtering guide
3. `DEPLOYMENT_CHECKLIST.md` - This file

---

## Deployment Commands (Copy-Paste Ready)

```powershell
# Navigate to project
cd "C:\vs code projects\MUTITIME-OI-STRATEGY"

# Verify all files present
ls src/fetch_india_vix.py
ls docs/VIX_FILTERING.md

# Test VIX
python src/fetch_india_vix.py

# Test strategy imports
python -c "from src.strategy import OpenInterestStrategy; print('✓ Strategy imports OK')"

# Syntax check
python -m py_compile src/strategy.py
python -m py_compile src/fetch_india_vix.py

# CRITICAL: Stop old scheduler
# Press Ctrl+C in scheduler terminal OR:
# taskkill /F /IM python.exe /FI "WINDOWTITLE eq *scheduler*"

# Start fresh scheduler
python src/oi_strategy_scheduler.py
```

---

## Risk Assessment

### Without These Fixes:
- **High Risk:** LTP mixup can occur again (211.85 scenario)
- **Medium Risk:** Trading in sideways market (VIX < 12)
- **Low Risk:** Unsubscribe errors (non-critical)

### With These Fixes:
- **Spike filter:** Eliminates bad tick entries
- **VIX filter:** Avoids unfavorable market conditions
- **Combined:** Multi-layer protection

---

## Contact Points

If issues occur tomorrow:
1. Check `logs/oi_scheduler.log` for VIX and SPIKE messages
2. Check `logs/trade_history.csv` for entry prices
3. Verify strategy.py is running latest code (check file timestamp)

---

## Confidence Level

**Spike Filter:** 🟢 High (simple validation logic)
**VIX Filter:** 🟢 High (tested, NSE API working)
**DataFrame Fix:** 🟡 Medium (depends on execution flow)
**Unsubscribe Fix:** 🟢 High (try-except wrapper)

**Overall Readiness:** 🟢 Ready for deployment

---

*Prepared: November 20, 2025 @ 11:35 PM IST*
*Deploy before: November 21, 2025 @ 9:15 AM IST*
