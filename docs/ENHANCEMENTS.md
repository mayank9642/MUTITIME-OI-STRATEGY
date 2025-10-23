# OI Strategy Enhancements Summary

## Implemented Enhancements

### 1. Fixed Symbol Format for Fyers API
- Updated the option symbol generation to follow Fyers API format: `NSE:NIFTY25JUN19500CE`
- Properly formatted expiry date strings in `nse_data.py`
- Added logging of sample symbols for verification

### 1A. Fixed Strike Price Extraction Bug (June 2025)
- Fixed a critical bug that was preventing the strategy from correctly parsing strike prices
- The error "Could not parse strike price from symbol" was resolved
- Implemented a robust regular expression pattern that reliably extracts the 5-digit strike price: `match = re.search(r'(\d{5})[CP]E$', symbol)`

### 2. Enhanced Market Hour Checks
- Added thorough checks for market open/close times
- Added weekend detection
- Added simple holiday detection
- Improved time-based logic for 9:20 AM analysis

### 3. Simplified Position Sizing for Paper Trading
- Set fixed position size of 1 lot for all trades
- Removed account equity checks for paper trading
- Streamlined trade execution process
- Added paper trade flag in trade history

### 4. Improved Trade Management
- Enhanced exit conditions tracking (stoploss, target, time-based)
- Added detailed P&L calculation for trades
- Created comprehensive trade history logging
- Fixed strike price extraction from symbols

### 5. Added Performance Tracking & Reporting
- Created daily summary reports
- Added CSV export of trade history
- Implemented end-of-day reporting
- Stored historical trade data

### 6. Created Web Dashboard
- Built interactive Dash application for monitoring
- Added real-time P&L chart
- Added trade history table
- Created daily statistics display
- Created convenience scripts to launch dashboard

### 7. Advanced Risk Management (June 2025)
- Added trailing stop loss that dynamically adjusts as trades move in favorable direction
- Configured via `use_trailing_stop`, `trailing_stop_pct`, and `trailing_trigger_pct` 
- Helps lock in profits while still allowing for maximum upside potential

### 8. Single Trade Per Day Limit (June 2025)
- Implemented a daily trade limit of one trade per day
- Uses the `trade_taken_today` flag to enforce this limit
- Trade flag resets at midnight, ensuring fresh trading opportunities each day

### 9. Real-Time P&L Monitoring (June 2025)
- Added second-by-second position updates after trade entry
- Continually displays entry price, current price, stop loss level, trailing stop, and P&L
- Implemented through a background thread for non-blocking operation

### 10. Trade Logging to Excel (June 2025)
- Added comprehensive Excel-based trade logs with professional formatting
- Each day gets its own Excel file with date stamp in the filename
- Includes columns for all relevant trade data: Entry/Exit DateTime, Symbol, Direction, Prices, Stop Loss, Target, P&L, etc.
- Enables better trade analysis and reporting capabilities

### 11. Strike Selection Filtering (June 2025)
- Added filters to avoid selecting strikes that are too far from current price
- Implemented `max_strike_distance` parameter (default: 500 points) to limit distance from ATM
- Increased minimum premium threshold to 50.0 (from default 3.0)
- Added sophisticated strike selection algorithm:
  1. First checks highest OI strikes from current expiry
  2. If premium < threshold, checks 2nd highest OI strikes
  3. If still below threshold, tries next expiry's highest OI strikes
  4. If still below threshold, tries 2nd highest OI from next expiry
- Ensures system always prioritizes highest OI (core strategy) while maintaining practical premium levels
- Logs detailed information about strike selection process and which criteria were used
- Prevents system from selecting deep OTM options with very low premiums
- Logs distance of selected strikes from current spot price for better decision-making

### 13. Performance Tracking System (June 2025)
- Added comprehensive trade tracking with detailed metrics for each trade
- Records statistics to `logs/trade_performance.csv` for future analysis
- Calculates win rate, average win/loss, and trade expectancy

### 14. Daily Trade Limit (June 2025)
- Added strict enforcement of one trade per day
- Prevents re-entry after a trade is closed to avoid overtrading
- Strategy will skip breakout monitoring after the daily limit is reached
- Logs clear messages when the daily limit is reached
- Trade limit resets at the beginning of each trading day

### 15. Real-Time Position Monitoring (June 2025)
- Added second-by-second position updates in active trades
- Displays entry price, current price, stoploss level, and P&L in real-time
- Uses a dedicated monitoring thread for continuous updates
- Provides clearer and more consistent trade status information
- Makes it easier to track position performance throughout the trade duration

## GTT Order Management & Mutual Exclusivity (August 2025)

### Overview
This project now features robust GTT (Good Till Trigger) order management for options OI breakout strategies. The system can place GTT orders for both CE and PE strikes after breakout levels are finalized, and ensures only one side is traded by automatically cancelling the other GTT order when one is triggered.

### Key Features
- **GTT Order Placement:**
  - Places GTT orders for both CE and PE strikes at their respective breakout levels.
  - Each GTT order is tracked by a unique ID and managed by the `OrderManager` class.
- **Mutual Exclusivity:**
  - When one GTT order (e.g., CE) is triggered, the other (e.g., PE) is automatically cancelled.
  - This ensures only one trade is executed per breakout event, preventing double entry.
- **Order Lifecycle Management:**
  - Orders are monitored for trigger, expiry, and cancellation.
  - All order state transitions are logged for traceability.
- **Thread Safety:**
  - All order management operations are thread-safe, supporting concurrent price updates and order actions.
- **Paper & Live Trading Support:**
  - Fully supports paper trading mode for safe testing.
  - Stubs provided for broker API integration for live trading.
- **Logging:**
  - All GTT order events (placement, trigger, cancel, expiry, error) are logged in detail.
- **Unit Tested:**
  - Comprehensive unit tests for all GTT order management logic.

### Example Log Output
```
Placed GTT orders: CE=abc123, PE=def456
GTT order triggered: {'order_id': 'abc123', ...}
CE triggered, PE GTT order cancelled: def456
```

### How It Works
1. After OI analysis, GTT orders are placed for both CE and PE strikes.
2. The system monitors both orders in real time.
3. When one order is triggered, the other is cancelled automatically.
4. Only the triggered side proceeds to trade logic and position management.

See `src/order_manager.py` and `src/strategy.py` for implementation details.

## How to Use

### Running the Strategy
1. Run authentication: `.\run_auth.ps1` or `run_auth.bat`
2. Run the main strategy: `python src\main.py`

### Viewing Performance Dashboard
1. Run the dashboard: `.\run_dashboard.ps1` or `run_dashboard.bat`
2. Open your browser to: http://localhost:8050

## Advanced Simulation & Backtesting Features

### 1. Enhanced Simulation Capabilities
- Created `src/enhanced_simulation.py` with the ability to simulate strategy for any date/time
- Implemented multiple market scenario templates (bullish, bearish, volatile, range-bound)
- Added ability to simulate strategy across multiple time points in a day
- Created realistic option chain data generation with proper volatility and OI characteristics

### 2. Historical Data & Backtesting
- Added `src/backtest_strategy.py` for comprehensive backtesting across date ranges
- Implemented historical data fetching from Fyers API when available
- Added data caching to improve performance and reduce API calls
- Created performance metrics calculation and reporting

### 3. Testing & Debugging Tools
- Added scripts to test option chain fetching directly
- Created `src/fetch_option_oi.py` for real-time OI analysis
- Improved logging throughout the codebase

### How to Use Simulation & Backtesting

#### Enhanced Simulation
- Run `.\run_enhanced_simulation.ps1` or `run_enhanced_simulation.bat`
- Optionally specify date and time: `.\run_enhanced_simulation.ps1 -date "2023-04-25" -time "09:20"`
- For multiple time points: `.\run_enhanced_simulation.ps1 -date "2023-04-25" -multiple`

#### Backtesting
- Run `.\run_backtest.ps1` or `run_backtest.bat`
- Specify date range: `.\run_backtest.ps1 -start "2023-04-01" -end "2023-04-30"`
- Results are saved in `data/backtest/` directory

## Next Steps
1. Integrate with additional data sources for more accurate historical data
2. Implement additional technical indicators for entry/exit
3. Add SMS/email notifications for trade alerts
4. Create parameter optimization tools
5. Implement automatic exchange holiday calendar

## Files Modified/Added

### Core Files
- `src/nse_data_new.py`: Fixed option symbol formatting and Fyers API calls
- `src/strategy.py`: Enhanced trade management, reporting, and added simulation support
- `src/main.py`: Improved scheduling and error handling

### Simulation & Testing 
- Created `src/enhanced_simulation.py`: Advanced simulation capabilities
- Created `src/backtest_strategy.py`: Historical data backtesting
- Created `src/fetch_option_oi.py`: Tool to fetch and analyze current OI data
- Created `src/test_option_chain.py`: Testing tool for option chain API

### Scripts & Support Files
- Created `run_enhanced_simulation.ps1`/`.bat`: Scripts to run enhanced simulations
- Created `run_backtest.ps1`/`.bat`: Scripts to run backtests
- Updated `requirements.txt`: Added dependencies for simulation and backtesting
- Enhanced logging and created data directories for storing simulation results

### 12. Excel-based Trade Logging
- Added Excel file format for trade history in addition to CSV
- Each day's trade data is saved to a date-stamped Excel file (e.g., `trade_history_20250624.xlsx`)
- Enhanced trade records with comprehensive trading details:
  - Entry/Exit DateTime with full timestamp
  - Trading Direction (PUT/CALL)
  - Entry/Exit Price
  - Stop Loss, Target, and Trailing SL values
  - Brokerage cost estimation
  - Profit/Loss calculation
  - Margin requirement estimation
  - Percentage Gain/Loss metrics
- Implemented professional Excel formatting:
  - Bold, colored headers with center alignment
  - Currency formatting for monetary values
  - Percentage formatting for gain/loss metrics
  - Auto-adjusted column widths for optimal viewing
  - Color-coded conditional formatting
- Maintained CSV format for backward compatibility
- Added fallback mechanism to ensure data is never lost even if Excel writing fails
