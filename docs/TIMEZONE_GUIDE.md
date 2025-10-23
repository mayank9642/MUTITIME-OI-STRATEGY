# Running the OI Strategy with IST Time Zone

This guide explains how to run the OI Strategy when your local time differs from Indian Standard Time (IST).

## Time Zone Support

The OI Strategy is designed to trade on the Indian markets, which follow Indian Standard Time (IST, UTC+5:30). 
The strategy has been enhanced to automatically handle time zone differences, ensuring that:

1. Market hours are properly detected (9:15 AM - 3:30 PM IST)
2. The 9:20 AM analysis happens at the correct time
3. Trade timestamps are correctly recorded in IST
4. Reports and logs use IST timestamps

## How to Run the Strategy

### Step 1: Install Required Packages

First, ensure you have all the required packages installed:

```powershell
cd "c:\vs code projects\OI-strategy"
pip install -r requirements.txt
```

This will install the `pytz` package needed for time zone handling.

### Step 2: Authentication

Run the authentication script to get a valid Fyers API token:

```powershell
cd "c:\vs code projects\OI-strategy"
.\run_auth.ps1
```

This needs to be done once per trading day.

### Step 3: Run the Strategy

Start the main strategy:

```powershell
cd "c:\vs code projects\OI-strategy"
python src/main.py
```

The strategy will automatically:
- Convert your local time to IST
- Wait until 9:15 AM IST to start
- Perform analysis at 9:20 AM IST
- Execute trades based on IST market hours

### Step 4: Monitor Performance

Launch the dashboard to monitor trading performance:

```powershell
cd "c:\vs code projects\OI-strategy"
.\run_dashboard.ps1
```

Open your browser to http://localhost:8050 to view the dashboard.

## Verifying Time Zone Handling

You can verify proper time zone handling by checking the logs:

1. The strategy logs the current IST time at each check
2. Trade timestamps are recorded in IST
3. The daily reports use IST dates

This ensures the strategy works correctly regardless of your local time zone.

## Testing During Non-Market Hours

If you want to test outside of IST market hours (9:15 AM - 3:30 PM IST), you can temporarily modify 
the strategy to simulate market hours. This should only be done for testing purposes.
