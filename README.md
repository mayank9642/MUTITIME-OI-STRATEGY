# Open Interest Option Buying Strategy

## Overview
This project implements an intraday option buying strategy based on open interest data. The strategy analyzes the Nifty 50 option chain at market open (9:20 AM), identifies strikes with the highest open interest, and enters trades when premium prices breakout by 10% - indicating potential momentum.

## Strategy Details
- **Analysis Time**: 9:20 AM
- **Entry Condition**: 10% increase in premium price from 9:20 AM level
- **Stoploss**: 20% of entry premium
- **Target**: 1:2 risk-reward ratio (2x the risk amount)
- **Maximum Holding Period**: 30 minutes
- **Exit Conditions**: Stoploss hit, Target achieved, or 30-minute time limit

## Setup Instructions

### Prerequisites
- Python 3.8+
- Fyers Trading Account
- Fyers API credentials

### Installation
1. Clone the repository
```bash
git clone https://github.com/mayank9642/open-interest-strategy.git
cd open-interest-strategy
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Update the config file with your Fyers credentials
```bash
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your credentials
```

## Directory Structure

The project has been organized into the following structure:
- `src/` - Core source code files
- `tests/` - Test files
- `logs/` - Log files
- `docs/` - Documentation
- `config/` - Configuration files
- `archive/` - Archived old versions
- `backups/` - Backup files

See `DIRECTORY_STRUCTURE.md` for more details.

## Usage

### Authentication
Before running the strategy, you need to authenticate with Fyers:

```bash
# Windows
.\run_auth.bat

# PowerShell
.\run_auth.ps1

# Linux/Mac
python src/auth.py
```

### Run the Fixed Strategy
```bash
# Windows
.\run_fixed_strategy.bat

# PowerShell
.\run_fixed_strategy.ps1

# Directly via Python
python run_fixed_strategy.py
```

### Wait for Market Open
To run the strategy with automatic waiting for market open (instead of exiting):
```bash
# Windows
.\wait_for_market_open.bat

# PowerShell
.\wait_for_market_open.ps1

# Directly via Python
python wait_and_run_strategy.py
```

The fixed strategy includes improvements:
- Corrected option symbol formatting for Fyers API
- Enhanced WebSocket implementation with better error handling
- Fixed indentation issues in the codebase
- Improved logging to both console and file
- Ability to wait for market open instead of exiting

### View Dashboard
The strategy includes a web-based dashboard for monitoring performance:

```bash
# Windows
.\run_dashboard.bat

# PowerShell
.\run_dashboard.ps1
```
Open your browser to [http://localhost:8050](http://localhost:8050)

## Simulation & Backtesting

The strategy now includes advanced simulation and backtesting capabilities to test performance with historical or synthetic data.

### Enhanced Simulation
Run the strategy with simulated option chain data for any date and time:

```bash
# Windows
.\run_enhanced_simulation.bat "2023-04-25" "09:20"

# PowerShell
.\run_enhanced_simulation.ps1 -date "2023-04-25" -time "09:20"
```

To simulate multiple time points during a day:

```bash
# Windows
.\run_enhanced_simulation.bat "2023-04-25" "09:20" multiple

# PowerShell
.\run_enhanced_simulation.ps1 -date "2023-04-25" -multiple
```

### Backtesting
Backtest the strategy across a range of dates:

```bash
# Windows
.\run_backtest.bat "2023-04-01" "2023-04-30"

# PowerShell
.\run_backtest.ps1 -start "2023-04-01" -end "2023-04-30"
```

Results are saved in the `data/backtest/` directory with performance metrics.

### Option Chain Testing
To directly test the option chain fetching functionality:

```bash
# Windows
.\run_option_chain_test.bat

# PowerShell
python src\test_option_chain.py
```

To fetch and display current OI data:

```bash
# Windows
.\run_fetch_oi.bat

# PowerShell
python src\fetch_option_oi.py
```

## Logging
All logs are stored in the `logs` directory:
- `strategy.log` - Main strategy log
- `fyersApi.log` - Fyers API log
- `fyersRequests.log` - Fyers API request log

## Additional Information
For more detailed information, see:
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [ENHANCEMENTS.md](ENHANCEMENTS.md) - List of enhancements
- [TIMEZONE_GUIDE.md](TIMEZONE_GUIDE.md) - Guide for timezone handling

# OI-STRATEGY

This project uses API polling only for option data. WebSocket code and hybrid logic have been removed for simplicity and reliability.

## How it works
- Option data is polled at a fast interval (0.5s) using the Fyers API.
- No WebSocket is used for index or option data.

## Legacy files removed
- hybrid_market_data.py
- test_option_symbol_formats.py
- fyers_support_report.py
- run_option_symbol_test.bat/.ps1
- run_fyers_support_report.bat

## Main script
- test_websocket_fixed.py (now only polls option data)