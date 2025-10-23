# Quick Start Guide for OI-Strategy

## Overview
OI-Strategy is an automated trading system for options based on Open Interest analysis. The strategy identifies options with highest open interest and enters trades when premium prices breakout, indicating potential momentum.

## Setup Instructions

### Step 1: Setup the Environment
Run the setup script to create necessary directories, install dependencies, and verify your configuration:

```bash
python setup.py
```

### Step 2: Configure Fyers API Credentials
Edit the `config/config.yaml` file and replace the placeholder values with your actual Fyers API credentials:

```yaml
fyers:
  client_id: "YOUR_FYERS_CLIENT_ID"  # Replace with actual client ID
  secret_key: "YOUR_SECRET_KEY"      # Replace with actual secret key
  redirect_uri: "https://fessorpro.com/"
  access_token: ""                   # Will be generated automatically
  token_expiry: ""                   # Will be set automatically
  totp_key: ""                       # Optional: For TOTP-based auth
```

### Step 3: Generate Access Token
Run the authentication module to obtain an access token:

```bash
python -m src.auth
```

This will:
1. Open your browser to the Fyers login page
2. Ask you to authorize the application
3. Generate and save an access token

### Step 4: Run the Strategy
Start the automated trading strategy:

```bash
python -m src.main
```

The strategy will:
1. Initialize for the trading day
2. Analyze option chain data at 9:20 AM
3. Monitor for breakout opportunities
4. Execute trades according to the strategy rules

## Strategy Parameters
You can customize the strategy parameters in `config/config.yaml`:

```yaml
strategy:
  symbol: "NSE:NIFTY50-INDEX"        # Underlying index for options
  analysis_time: "09:20"             # Time to analyze option chain
  max_holding_minutes: 30            # Maximum position holding time
  risk_reward_ratio: 2               # Target is 2x the risk
  stoploss_pct: 20                   # Stoploss percentage of premium
  breakout_pct: 10                   # Premium increase needed for entry
```

## Troubleshooting
- If you encounter authentication issues, delete the access token and run `python -m src.auth` again
- Check the log files in the `logs` directory for detailed error information
- Ensure your Fyers account has sufficient funds and is authorized for options trading

## Advanced Features
- TOTP Authentication: If you have TOTP set up for your Fyers account, add your TOTP key to the config file
- Custom Order Types: The system supports market, limit, stop-loss, and stop-loss limit orders
- Real-time Data: WebSocket implementation for live market data
- Order Tracking: Full order management with status tracking

For more information, refer to the complete documentation in the README.md file.
