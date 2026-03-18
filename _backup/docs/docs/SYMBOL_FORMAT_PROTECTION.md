# Symbol Format Protection Guide

## Problem
Fyers periodically changes option symbol formats, causing websocket connection failures:
- **Format A**: `NSE:NIFTY25NOV26000CE` (full month name)
- **Format B**: `NSE:NIFTY25N26000CE` (abbreviated month code)

This typically happens during expiry rollovers or system updates.

## Protection Mechanisms

### 1. Automatic Format Retry (Already Active)
**Location**: `src/strategy.py` → `retry_websocket_connection()`

When websocket connection fails:
1. Tries original format (from OI data)
2. If fails, automatically tries alternate format
3. Each format gets 3 retry attempts
4. Seamless switching without manual intervention

**How it works:**
```python
# Original: NSE:NIFTY25NOV26000CE
# If fails, tries: NSE:NIFTY25N26000CE
# OR vice versa
```

### 2. Symbol Format Validation (New)
**Location**: `src/symbol_validator.py`

**Purpose**: Test symbol format BEFORE live trading to catch issues early

**Usage:**

#### Option A: Manual Pre-Trading Check
```powershell
# Run before market open or after expiry change
python check_symbol_format.py
```

**Output:**
```
✓ PASS: Strategy should work with current symbol format
Format Type: FULL
Pattern: NIFTY + day + MONTH + strike + CE/PE
Example: NSE:NIFTY25NOV26000CE
```

#### Option B: Automatic Daily Check
**Location**: `src/oi_strategy_scheduler.py`

Runs automatically at 9:15 AM market open:
```
SYMBOL FORMAT VALIDATION CHECK
✓ Symbol validation PASSED: Full month format (NOV, DEC) is working
```

### 3. Symbol Format Variants
**Location**: `src/symbol_formatter.py` → `get_symbol_variants()`

Generates all possible formats for a symbol:
```python
Input:  NSE:NIFTY25NOV26000CE
Output: [
    'NSE:NIFTY25NOV26000CE',  # Full month
    'NSE:NIFTY25N26000CE'      # Abbreviated
]
```

## When Symbol Format Changes

### Symptoms:
- Websocket errors: "Please provide a valid symbol"
- `price=None` in logs
- Repeated connection failures

### Automatic Response:
1. ✅ Strategy detects subscription failure
2. ✅ Automatically tries alternate format
3. ✅ Connects successfully with working format
4. ✅ Trading continues normally

### If Automatic Fix Fails:
1. **Run manual validation:**
   ```powershell
   python check_symbol_format.py
   ```

2. **Check logs:**
   ```
   logs/symbol_validation.log
   logs/strategy.log (search for "VALIDATION")
   ```

3. **Look for errors:**
   - "invalid_symbols" in websocket errors
   - Symbol format mismatches

4. **Emergency fix** (if needed):
   - Fyers may have introduced NEW format
   - Update `src/symbol_utils.py` → `convert_option_symbol_format()`
   - Add new format to `get_symbol_variants()`

## Monitoring Symbol Format Health

### Daily Checklist:
1. ✅ Check scheduler logs at market open for validation result
2. ✅ Verify "WebSocket connected successfully" appears
3. ✅ Confirm prices are flowing (not `price=None`)

### Weekly Checklist (Especially on Expiry Week):
1. ✅ Run `python check_symbol_format.py` on Thursday/Friday
2. ✅ Check if format changed compared to previous week
3. ✅ Monitor first trade after rollover carefully

### Monthly Checklist:
1. ✅ Review `logs/symbol_validation.log` for patterns
2. ✅ Note which format Fyers is using each month
3. ✅ Update this document if new formats appear

## Symbol Format Reference

### Current Known Formats:

#### Format 1: Full Month Names (Most Common)
```
Pattern: NSE:NIFTY + DD + MMM + STRIKE + TYPE
Example: NSE:NIFTY25NOV26000CE
         NSE:NIFTY02DEC25900PE

Months: JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC
```

#### Format 2: Abbreviated Month Codes (Alternate)
```
Pattern: NSE:NIFTY + DD + M + STRIKE + TYPE
Example: NSE:NIFTY25N26000CE
         NSE:NIFTY02D25900PE

Month Codes:
JAN=A, FEB=B, MAR=C, APR=D, MAY=E, JUN=F,
JUL=G, AUG=H, SEP=I, OCT=J, NOV=N, DEC=O
```

## Files Modified for Protection

1. **`src/symbol_utils.py`**: Core conversion logic (removed year component)
2. **`src/symbol_formatter.py`**: Variant generation
3. **`src/symbol_validator.py`**: Format testing (NEW)
4. **`src/strategy.py`**: Retry logic with variants
5. **`src/oi_strategy_scheduler.py`**: Daily validation check
6. **`check_symbol_format.py`**: Manual validation script (NEW)

## Testing Symbol Format Changes

### Simulate Format Change:
1. Stop strategy
2. Manually test connection:
   ```python
   from src.symbol_validator import validate_symbol_format
   result = validate_symbol_format()
   print(result)  # Should be 'full' or 'abbreviated'
   ```

3. Force specific format:
   ```python
   from src.fixed_improved_websocket import improved_market_data_websocket
   
   # Test full format
   client1 = improved_market_data_websocket(['NSE:NIFTY25NOV26000CE'])
   
   # Test abbreviated format
   client2 = improved_market_data_websocket(['NSE:NIFTY25N26000CE'])
   ```

## Emergency Contacts / Resources

1. **Fyers API Documentation**: Check for symbol format updates
2. **Fyers Support**: Report if new format not documented
3. **This Strategy's Logs**: 
   - `logs/strategy.log`
   - `logs/symbol_validation.log`
   - `logs/oi_scheduler.log`

## Version History

- **2025-11-19**: Added symbol format validation and protection
  - Fixed year component removal
  - Added automatic retry with format variants
  - Added daily validation at market open
  - Created manual validation script

## Summary

✅ **You're Protected**: The strategy now has 3 layers of protection
✅ **Automatic**: Format retry happens without your intervention  
✅ **Proactive**: Daily validation warns you before trading
✅ **Manual**: You can check format anytime with validation script

**Most importantly**: Even if Fyers changes format on next expiry, the automatic retry logic will detect and switch formats, keeping your strategy running!
