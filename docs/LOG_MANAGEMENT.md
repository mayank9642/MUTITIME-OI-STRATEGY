# Log Management Guide

This document provides instructions for managing logs in the trading strategy application to ensure sensitive information is not exposed and log files remain manageable in size.

## Overview

The trading strategy application has been updated to reduce unnecessary debug logging, specifically:

1. Removed debug logs that expose sensitive authentication token information
2. Reduced repetitive option chain data structure logging that appears frequently in log files
3. Added tools to sanitize log files and prevent further exposure of sensitive data

## Log Filter Implementation

Two main mechanisms have been implemented to filter logs:

1. **Runtime Filtering**: A `SensitiveInfoFilter` class in `main.py` prevents sensitive information from being logged in the first place.
2. **Post-processing**: The `log_sanitizer.py` script can clean existing log files by removing or redacting sensitive information.

## How to Use the Log Sanitizer

### One-time Cleanup

To sanitize all existing log files:

```
python src/log_sanitizer.py
```

### Continuous Monitoring

To start a background monitor that continuously sanitizes logs:

```
python src/log_sanitizer.py --monitor
```

### Scheduled Task

For regular maintenance, set up a scheduled task to run the sanitization:

#### Windows

1. Open Task Scheduler
2. Click "Create Basic Task"
3. Name the task "Trading Strategy Log Sanitization"
4. Set the trigger to run daily (or at your preferred frequency)
5. Select "Start a program" as the action
6. Browse to select the `sanitize_logs.bat` script
7. Set the "Start in" field to the application directory
8. Complete the wizard

Or run this PowerShell command as Administrator:

```powershell
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$(Get-Location)\sanitize_logs.bat`"" -WorkingDirectory "$(Get-Location)"
$trigger = New-ScheduledTaskTrigger -Daily -At 00:00
$settings = New-ScheduledTaskSettingsSet -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "Trading Strategy Log Sanitization" -Action $action -Trigger $trigger -Settings $settings -Description "Sanitize trading strategy logs daily"
```

## Log Files Affected

The sanitization process affects the following files:

- `logs/strategy.log`: The main strategy log file
- All backup log files in the `logs/` directory

## What Information Gets Filtered

1. **Authentication Information**:
   - API keys, access tokens, and client IDs
   - Debug messages from `get_fyers_client` function

2. **Option Chain Data Structure**:
   - Repetitive "Sample option data structure" messages
   - Option data structure field dumps

## Technical Implementation

- The `SensitiveInfoFilter` class in `main.py` filters logs at runtime
- The `nse_data_new.py` module has been updated to remove excessive option chain data structure logging
- The `log_sanitizer.py` script provides both immediate cleanup and continuous monitoring

## Monitoring and Verification

To verify that log filtering is working correctly, run:

```
python src/test_log_filtering.py
```

This test generates sample logs with sensitive information and verifies that they're properly filtered.
