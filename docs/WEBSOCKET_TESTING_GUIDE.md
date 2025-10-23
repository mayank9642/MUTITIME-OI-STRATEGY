# Websocket Price Updates: Testing and Verification Guide

This guide explains how to test and verify that the websocket price updates are working correctly, 
especially with the enhanced market-closed handling capabilities.

## Testing Tools

1. **Websocket Health Check**: `run_websocket_health_check.bat` or `run_websocket_health_check.ps1`
   - This performs a comprehensive diagnostic check of the websocket connection
   - Reports what types of data are being received for each symbol
   - Clearly identifies whether the market is open or closed

2. **Websocket Price Updates Test**: `run_websocket_test.bat` or `run_websocket_test.ps1`
   - Tests the websocket price updates with enhanced logging
   - Shows detailed information about the data being received
   - Provides diagnostics about market open/closed status

## What to Check When Market is Open

When the market is open, you should see:

1. Real-time price updates with `ltp` (Last Traded Price) values
2. WebSocket update logs showing price changes
3. The health check should report `Status: OK` and show `has_ltp: true` for symbols

Example log when market is open:
```
WebSocket update: NSE:NIFTY25JUL24700PE price updated to 75.45
WebSocket update: NSE:NIFTY25JUL24700PE price updated to 75.50
WebSocket update: NSE:NIFTY25JUL24800CE price updated to 117.95
```

## What to Check When Market is Closed

When the market is closed, you should see:

1. Market closed warning messages in logs
2. Price updates using `prev_close_price` instead of `ltp`
3. The health check should report `Status: WARNING` with message indicating market is closed
4. Fields reported will include `prev_close_price` but not `ltp`

Example log when market is closed:
```
MARKET IS CLOSED - Using previous close for NSE:NIFTY25JUL24700PE: 75.4
MARKET STATUS: Receiving websocket data but no price updates - MARKET IS LIKELY CLOSED
```

## Verifying the Fix

To verify that the fixes are working correctly:

1. **During Market Hours (9:15 AM - 3:30 PM IST)**:
   - Run the tests and confirm real-time price updates are received
   - Prices should change frequently reflecting market movements

2. **Outside Market Hours**:
   - Run the tests and confirm the market closed state is properly detected
   - The system should use previous closing prices rather than showing null values
   - All diagnostics should clearly indicate market is closed

## Troubleshooting

If you're still experiencing issues:

1. Check the logs for error messages related to websocket connections
2. Verify your Fyers API authentication is current and valid
3. Confirm internet connectivity is stable
4. Try running the self-diagnostic with `python check_strategy.py`

The most important improvement in this update is the ability to clearly distinguish between 
market closed conditions (where prices legitimately don't update) and actual websocket failures.
