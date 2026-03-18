# VIX Forward Testing Guide

## Overview

Instead of relying solely on expert VIX thresholds, this forward testing approach allows you to **discover the optimal VIX range for YOUR specific strategy** based on actual performance data.

## How It Works

### Phase 1: Data Collection (2-4 weeks)

1. **VIX filtering is DISABLED** - Strategy trades normally across all VIX conditions
2. **VIX data is LOGGED** at every strategy run to `logs/vix_history.csv`
3. **Every trade is recorded** with its corresponding VIX value and outcome
4. **No trades are blocked** - you collect performance data across all VIX ranges

### Phase 2: Analysis

After collecting sufficient data (30-50+ trades):

```powershell
python analyze_vix_data.py
```

This generates a comprehensive report showing:
- Win rate by VIX range (VIX <10, 10-15, 15-20, 20-25, 25-30, 30-35, >35)
- Average P&L by VIX range
- Total P&L by VIX range
- Best and worst performing VIX ranges for YOUR strategy

### Phase 3: Optimization

Based on the analysis, you'll get:
- **Suggested VIX thresholds** tailored to your strategy
- **Ranges to avoid** (poor win rate or negative P&L)
- **Optimal trading conditions** backed by your own data

### Phase 4: Enable Filtering

Once confident in the data:
1. Update `config.yaml` with suggested VIX thresholds
2. Enable `vix_check_enabled: true`
3. Strategy will now only trade during optimal VIX conditions
4. Keep `vix_logging_enabled: true` to continue refining

## Configuration

### Current Setup (Forward Testing Mode)

```yaml
strategy:
  # FORWARD TESTING MODE: Collecting data across all VIX ranges
  vix_check_enabled: false          # Filtering disabled - trades all VIX conditions
  vix_logging_enabled: true         # Logging enabled - records all VIX data
  vix_min_threshold: 15.0           # Reference only (not enforced)
  vix_max_threshold: 35.0           # Reference only (not enforced)
```

### After Optimization (Production Mode)

```yaml
strategy:
  # PRODUCTION MODE: Trading only optimal VIX ranges
  vix_check_enabled: true           # Filtering enabled based on your data
  vix_logging_enabled: true         # Keep logging to refine over time
  vix_min_threshold: 16.0           # Example: Your optimal minimum
  vix_max_threshold: 28.0           # Example: Your optimal maximum
```

## VIX History Data Structure

`logs/vix_history.csv` contains:

| Field | Description |
|-------|-------------|
| `timestamp` | Full datetime of strategy run |
| `date` | Date only |
| `time` | Time only |
| `weekday` | Day of week |
| `india_vix` | Current India VIX value |
| `vix_condition` | VIX classification (e.g., "Normal", "Choppy") |
| `vix_risk_level` | Risk assessment |
| `should_trade_per_vix` | Whether VIX was in "safe" range |
| `vix_recommendation` | VIX-based recommendation |
| `oi_analysis_done` | Whether OI analysis completed |
| `trade_signal` | LONG/SHORT or blank |
| `trade_executed` | True if trade taken |
| `entry_price` | Entry price if traded |
| `exit_price` | Exit price if traded |
| `pnl` | Trade P&L if traded |
| `pnl_percent` | Trade P&L percentage |
| `outcome` | WIN/LOSS/NO_TRADE/etc |
| `notes` | Additional context |

## Analysis Output Example

```
================================================================================
VIX FORWARD TESTING REPORT
================================================================================
Total Strategy Runs: 47
Total Trades Executed: 28
Overall Win Rate: 57.14%
Overall Avg P&L: ₹324.50
Overall Total P&L: ₹9,086.00

--------------------------------------------------------------------------------
PERFORMANCE BY VIX RANGE
--------------------------------------------------------------------------------
VIX Range                 Trades   Win%     Avg P&L      Total P&L    Avg VIX   
--------------------------------------------------------------------------------
VIX <10 (Extreme Low)     2        0.0      ₹-850.00     ₹-1,700.00   9.20      
VIX 10-15 (Low/Choppy)    8        37.5     ₹-142.50     ₹-1,140.00   12.80     
VIX 15-20 (Normal)        12       66.7     ₹485.00      ₹5,820.00    17.40     
VIX 20-25 (Elevated)      4        75.0     ₹920.00      ₹3,680.00    22.10     
VIX 25-30 (High)          2        50.0     ₹713.00      ₹1,426.00    27.50     

--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------
✓ Best Win Rate: VIX 20-25 (Elevated) - 75.0% (4 trades)
✓ Best Total P&L: VIX 15-20 (Normal) - ₹5,820.00 (12 trades)
✗ Worst Win Rate: VIX <10 (Extreme Low) - 0.0% (2 trades)
✗ Worst Total P&L: VIX 10-15 (Low/Choppy) - ₹-1,140.00 (8 trades)

--------------------------------------------------------------------------------
SUGGESTED CONFIG.YAML SETTINGS:
--------------------------------------------------------------------------------
vix_min_threshold: 15.4
vix_max_threshold: 29.5
vix_check_enabled: true
vix_logging_enabled: true  # Keep logging to refine over time
--------------------------------------------------------------------------------
```

## Benefits of This Approach

### 1. **Data-Driven Decisions**
- Based on YOUR strategy's actual performance
- Not generic expert opinions
- Accounts for your specific entry/exit logic

### 2. **No Missed Opportunities**
- During testing, you don't miss trades
- Collect comprehensive data across all conditions
- Can still profit while gathering data

### 3. **Continuous Improvement**
- Keep logging even after optimization
- Refine thresholds as market conditions evolve
- Adapt to changing volatility regimes

### 4. **Risk Management**
- Avoid trading in YOUR strategy's worst VIX conditions
- Focus capital on YOUR best-performing conditions
- Improve overall win rate and expectancy

## Recommended Timeline

### Week 1-2: Initial Collection
- **Goal**: 15-20 trades minimum
- **Action**: Trade normally, let VIX logger collect data
- **Analysis**: Run `analyze_vix_data.py` to check progress

### Week 3-4: Pattern Recognition
- **Goal**: 30-50 trades total
- **Action**: Continue collecting, look for consistent patterns
- **Analysis**: Identify clearly profitable vs unprofitable VIX ranges

### Week 5+: Implementation
- **Goal**: High confidence in optimal range
- **Action**: Enable VIX filtering with your thresholds
- **Monitoring**: Continue logging, refine monthly

## Important Notes

### Sample Size Requirements

- **Minimum 3 trades per VIX range** for statistical relevance
- **Minimum 30 total trades** for overall confidence
- **50+ trades ideal** for robust optimization

### Market Condition Changes

- VIX characteristics can shift over months/years
- Re-analyze quarterly or after major market events
- Keep `vix_logging_enabled: true` always

### False Signals

- Some VIX ranges may show good performance due to luck
- Look for consistency: high win rate AND positive total P&L
- Prefer ranges with more trade samples

## Viewing Raw Data

To examine raw VIX logs:

```powershell
# View in Excel
start logs\vix_history.csv

# View in Python
import pandas as pd
df = pd.read_csv('logs/vix_history.csv')
print(df)
```

## Troubleshooting

### "No VIX history data available"
- VIX logging hasn't started yet
- Check `vix_logging_enabled: true` in config
- Restart scheduler to load new config

### "Insufficient data"
- Keep trading to collect more data
- Minimum 10 records needed for basic analysis
- 30+ trades recommended for optimization

### "No consistently profitable ranges yet"
- Need more trades per VIX range
- Continue for 1-2 more weeks
- May indicate strategy needs other improvements

## Expert VIX Ranges (For Reference)

While you develop your own thresholds, these are expert guidelines:

| VIX Range | Market Condition | Expert Recommendation |
|-----------|------------------|----------------------|
| < 10 | Extreme Complacency | Avoid - Low volatility, false breakouts |
| 10-15 | Low Volatility (Choppy) | Caution - Sideways markets common |
| 15-20 | Normal (Sweet Spot) | Trade - Optimal conditions |
| 20-25 | Elevated | Trade - Good trending moves |
| 25-30 | High Volatility | Caution - Increased risk |
| 30-35 | Very High | High Risk - Whipsaws common |
| > 35 | Extreme Panic | Avoid - Crisis conditions |

**Your data may differ!** That's why forward testing is valuable.

## Next Steps

1. **Start Collecting**: Scheduler is configured for forward testing
2. **Trade Normally**: All VIX conditions will be tested
3. **Weekly Check**: Run `analyze_vix_data.py` every Friday
4. **Optimize**: After 30+ trades, implement suggested thresholds
5. **Monitor**: Keep logging and refine quarterly

---

**Remember**: The goal is to find YOUR strategy's optimal VIX range through real performance data, not assumptions.
