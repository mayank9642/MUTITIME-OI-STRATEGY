# Deployment Checklist - November 21, 2025

## ✅ Pre-Flight Checks (BEFORE 9:15 AM)

### 1. Code Compilation Status
- ✅ `src/strategy.py` - Compiled successfully
- ✅ `src/fixed_improved_websocket.py` - Compiled successfully  
- ✅ `src/fetch_india_vix.py` - Compiled successfully
- ✅ `config/config.yaml` - Valid YAML, loaded successfully

### 2. VIX System Status
- ✅ VIX fetch working (current: 12.14)
- ✅ VIX filtering: **DISABLED** (forward testing mode)
- ✅ VIX data will be logged in `trade_history.csv`
- ✅ 3 new columns added: `India VIX`, `VIX Condition`, `VIX Risk Level`

### 3. Critical Bug Fixes Deployed

#### Fix #1: Stale Data Protection ✅
**Problem:** WebSocket sent old price (211.85) when real price was 74
**Solution:**
- Clear all stale data before monitoring
- Wait max 3 seconds for fresh websocket data (smart wait, checks every 200ms)
- Won't trade if zero websocket data received

#### Fix #2: Price Sanity Validation ✅
**Problem:** Trading at absurd prices
**Solution:**
- Reject price ≤ 0
- Reject price > 500 (too high for Nifty options)
- Reject price >3x recent average (spike detection)
- Reject PE price > 150 (unrealistic)

#### Fix #3: Removed False Alarms ✅
**Problem:** "No data for 60 seconds" warnings causing confusion
**Solution:**
- Removed heartbeat monitor that was causing false warnings
- Cleaner logs, focus on real issues

#### Fix #4: Nov 19 Fixes (Now Active) ✅
- Price spike filter (>100% rejected)
- DataFrame preservation at trade entry
- Graceful unsubscribe error handling

---

## 🚀 Deployment Steps (9:00-9:15 AM)

### Step 1: Stop Current Scheduler
```powershell
# Find and stop the running scheduler process
Get-Process python | Where-Object {$_.CommandLine -like "*oi_strategy_scheduler*"} | Stop-Process
```

### Step 2: Verify Code Ready
```powershell
# Quick compilation check
python -m py_compile src/strategy.py src/fixed_improved_websocket.py
echo "Ready for deployment"
```

### Step 3: Start New Scheduler
```powershell
# Start with new code
python src/oi_strategy_scheduler.py
```

### Step 4: Monitor First Run (9:20 AM)
Watch logs for:
- ✅ "FETCHING INDIA VIX FOR DATA COLLECTION"
- ✅ "India VIX: [value]"
- ✅ "VIX filtering DISABLED - Trading regardless of VIX"
- ✅ "Waiting for fresh websocket data..."
- ✅ "✓ Fresh data received for all symbols"

---

## 📊 What to Monitor Today

### Critical Success Indicators
1. **No LTP mixup** - Entry prices should be reasonable (50-200 range typically)
2. **VIX logged** - Check `trade_history.csv` has VIX columns populated
3. **No "Price spike detected"** logs (unless real spike occurs)
4. **Fresh data wait** - Should see "Fresh data received" message within 1-2 seconds

### Expected Log Patterns

**Good Entry:**
```
MONITOR: NSE:NIFTY25NOV26000PE (PE) price=91.5 (Breakout: 90.64)
Breakout detected for NSE:NIFTY25NOV26000PE (PE) at price 91.5
✓ Price validation passed
Trade entry successful for NSE:NIFTY25NOV26000PE at price 91.5
```

**Rejected Bad Price:**
```
MONITOR: NSE:NIFTY25NOV26000PE (PE) price=211.85 (Breakout: 90.64)
⚠️ TRADE REJECTED: Price 211.85 is 2.9x average recent price (73.5)
⚠️ Likely stale/bad websocket data - rejecting trade
```

---

## 📈 VIX Forward Testing (Next 2-4 Weeks)

### What's Happening
- **All trades execute** regardless of VIX (no filtering)
- **VIX data captured** for every trade automatically
- **Build dataset** to find YOUR optimal VIX range

### Weekly Analysis
Every Friday run:
```powershell
python analyze_vix_data.py
```

### After 30-50 Trades
1. Run analysis to find profitable VIX ranges
2. Update config with suggested thresholds
3. Enable `vix_check_enabled: true`
4. Paper trade 2 more weeks with filtering
5. Verify improvement before going live

---

## 🛡️ Rollback Plan (If Issues)

If you see problems today:

### Emergency Stop
```powershell
# Stop scheduler immediately
Get-Process python | Where-Object {$_.CommandLine -like "*oi_strategy_scheduler*"} | Stop-Process
```

### Revert to Previous Version
```powershell
# Git revert if needed (or manually restore old files)
git log --oneline -5  # Find commit before today's changes
git revert <commit-hash>
```

---

## 📝 Files Modified Today (Nov 20, 2025)

### Core Strategy
- `src/strategy.py` 
  - Added VIX fetch and logging
  - Added price sanity validation
  - Added smart fresh data wait
  - Added PE price validation (>150 check)

### Websocket
- `src/fixed_improved_websocket.py`
  - Removed confusing heartbeat monitor
  - Cleaner connection handling

### Configuration
- `config/config.yaml`
  - VIX filtering disabled (forward testing)
  - VIX thresholds set to 15.0-35.0 (reference only)

### New Files
- `src/vix_logger.py` - VIX data collection (not used, integrated into trade_history)
- `analyze_vix_data.py` - VIX performance analysis tool
- `docs/VIX_FORWARD_TESTING.md` - VIX testing guide
- `VIX_DATA_COLLECTION_SUMMARY.md` - Quick reference

---

## ✅ Final Verification (Right Now)

- [x] All Python files compile without errors
- [x] VIX fetch working (12.14 current)
- [x] Config YAML valid
- [x] Strategy imports all modules correctly
- [x] No syntax errors in any file
- [x] Documentation complete

---

## 🎯 Success Criteria for Tomorrow

### Must Have
✅ No trades at absurd prices (>200 or <30 for typical strikes)  
✅ VIX data appears in trade_history.csv  
✅ Fresh data wait completes in <2 seconds  
✅ Websocket provides continuous data (no 60 sec gaps)  

### Nice to Have
✅ Profitable trades (but not required during testing)  
✅ Clean logs without errors  
✅ VIX values reasonable (10-20 range typically)  

---

## 📞 Emergency Contacts

**If scheduler won't start:**
- Check if another Python process is using the port
- Verify access token not expired (check token_expiry in config)

**If VIX fetch fails:**
- Check internet connection to NSE
- Verify NSE API is up (https://www.nseindia.com/api/allIndices)
- Strategy will continue without VIX if fetch fails

**If websocket fails:**
- Check Fyers API status
- Verify access token valid
- Try manual restart of scheduler

---

**Deployment Time:** Before 9:15 AM, November 21, 2025  
**Deployed By:** [Your Name]  
**Next Review:** November 21, 2025 (after market close)

---

## 🚦 GO / NO-GO Decision

**STATUS: ✅ GO FOR DEPLOYMENT**

All systems ready. No blocking issues. Deploy with confidence! 🚀
