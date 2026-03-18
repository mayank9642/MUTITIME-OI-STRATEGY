# Strike Prices and Times for Verifying Slippage on Chart

## Trade 1: NIFTY 25950 CE (Call Option)
**Strike Price:** 25950  
**Expiry:** 25-NOV-2025 (Weekly expiry - November 25, 2025)

### Timeline:
- **13:30:14** - Entry at ₹117.8
- **13:30:16** - Exit at ₹74.2 (2 seconds later!)

### What the logs show:
```
13:30:XX - Price oscillating: 73.4 → 117.0 → 118.2 → 119.05 → 117.8 (ENTRY)
13:30:XX - Then crashed: 117.8 → 74.2 (EXIT)
```

### To verify on chart:
1. Open NIFTY 25-NOV-2025 25950 CE option chart
2. Go to 13:30 PM on November 25, 2025
3. Use 1-second or tick-by-tick timeframe
4. Look for: 
   - Spike to ₹117-119 range
   - Immediate crash back to ₹74 range

**Expected finding:** This was likely a **data spike/glitch** or very brief spike that reverted immediately.

---

## Trade 2: NIFTY 26050 PE (Put Option)  
**Strike Price:** 26050
**Expiry:** 25-NOV-2025 (Weekly expiry - November 25, 2025)

### Timeline:
- **14:30:12** - Entry at ₹92.55
- **14:30:18** - Exit at ₹47.8 (6 seconds later!)

### To verify on chart:
1. Open NIFTY 25-NOV-2025 26050 PE option chart
2. Go to 14:30 PM (2:30 PM) on November 25, 2025
3. Use 1-second or tick-by-tick timeframe
4. Look for:
   - Entry around ₹92-93
   - Crash to ₹47-48 within seconds

**Expected finding:** Either data spike or genuine crash from ₹92 to ₹47 (-48% in 6 seconds).

---

## Trade 3: NIFTY 25950 PE (Put Option)
**Strike Price:** 25950
**Expiry:** 25-NOV-2025 (Weekly expiry - November 25, 2025)

### Timeline:
- **15:15:12** - Entry at ₹114.4
- **15:15:14** - Exit at ₹64.6 (2 seconds later!)

### To verify on chart:
1. Open NIFTY 25-NOV-2025 25950 PE option chart
2. Go to 15:15 PM (3:15 PM) on November 25, 2025
3. Use 1-second or tick-by-tick timeframe
4. Look for:
   - Entry around ₹114
   - Drop to ₹64-65 within 2 seconds

**Expected finding:** Sharp drop from ₹114 to ₹64 (-43% in 2 seconds).

---

## Summary Table

| Trade | Strike | Type | Entry Time | Entry Price | Exit Time | Exit Price | Duration | Drop |
|-------|--------|------|------------|-------------|-----------|------------|----------|------|
| 1 | 25950 | CE | 13:30:14 | ₹117.8 | 13:30:16 | ₹74.2 | **2 sec** | -37% |
| 2 | 26050 | PE | 14:30:12 | ₹92.55 | 14:30:18 | ₹47.8 | **6 sec** | -48% |
| 3 | 25950 | PE | 15:15:12 | ₹114.4 | 15:15:14 | ₹64.6 | **2 sec** | -43% |

---

## Important Notes for Chart Analysis

### 1. These are NIFTY Weekly Options (25-NOV expiry)
- Expired on the same day (November 25, 2025)
- 0DTE (Zero Days To Expiry) options are EXTREMELY volatile
- Price swings of 30-50% in seconds are possible near expiry

### 2. Time Context
All three trades happened in the **LAST 2.5 HOURS** of trading (13:30 to 15:15):
- Market closes at 15:30
- Options expire at 15:30
- This is when options can have wild swings

### 3. What to Look For on Charts

**If these were real price moves:**
- You'll see the price action clearly on 1-min or 5-min charts
- Volume bars will show activity
- Nearby strikes will show similar patterns

**If these were data spikes/glitches:**
- Price will show a "needle" spike that immediately reverts
- Volume may not match the move
- Nearby strikes won't show the same pattern
- The price at entry (₹117.8 for 25950CE) seems too high if spot was stable

### 4. Trade 1 Analysis (25950 CE)
The logs show something suspicious:
```
Previous: 73.65 | New: 117.8 | Change: 59.9% (ENTRY)
Previous: 117.8 | Current: 74.2 | Change: -37.01% (EXIT - 2 seconds later!)
```

This pattern suggests:
- **Possible data spike:** Price was ₹73-74, spiked to ₹117, immediately back to ₹74
- Our strategy bought at the spike (₹117.8) 
- Exited when it reverted to real price (₹74.2)

---

## Recommendation

### Check on NSE/Fyers historical data:
1. Go to Fyers/Tradingview/NSE website
2. Search for: **NIFTY25NOV25950CE**, **NIFTY25NOV26050PE**, **NIFTY25NOV25950PE**
3. View 1-minute candles for November 25, 2025 between 13:30-15:30
4. Verify if these price levels actually traded or were data glitches

### If these were real prices:
✅ The SL exit price fix is CRITICAL (would have saved ₹3,699)

### If these were data spikes:
✅ Need to add **spike rejection** for entry signals too (not just monitoring)  
✅ Current code rejects spikes during monitoring but accepts them during breakout detection

---

## Chart Verification Commands

### For Fyers Desktop:
1. Open chart for `NSE:NIFTY25NOV25950CE`
2. Set timeframe to 1-minute
3. Navigate to 25-Nov-2025, 1:30 PM
4. Check if ₹117.8 actually traded

### For TradingView:
1. Symbol: `NIFTY25NOV25950CE` (if available)
2. Or check NIFTY spot movement at those times
3. If NIFTY dropped 500+ points in 2 seconds → Options would crash
4. If NIFTY was stable → These were bad data ticks

---

**Created:** November 26, 2025  
**Purpose:** Verify whether massive option price drops were real or data glitches
