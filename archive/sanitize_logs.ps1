# sanitize_logs.ps1
# Script to sanitize log files by removing sensitive information
# Run this script periodically or as a scheduled task

# Add timestamp to console output
function Write-TimestampedMessage {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "$timestamp - $Message"
}

# Run the sanitizer on all logs
Write-TimestampedMessage "Starting log sanitization process..."
$startTime = Get-Date

try {
    # Change to script directory
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $scriptDir
    
    # Run the log sanitizer
    Write-TimestampedMessage "Running log sanitizer script..."
    python src/log_sanitizer.py
    
    # Check result
    if ($LASTEXITCODE -eq 0) {
        Write-TimestampedMessage "Log sanitization completed successfully."
    } else {
        Write-TimestampedMessage "Log sanitization encountered issues. Exit code: $LASTEXITCODE"
    }
    
    # Also run the log file monitor if it's not already running
    $monitorProcesses = Get-WmiObject Win32_Process -Filter "CommandLine LIKE '%log_sanitizer.py --monitor%'"
    if ($monitorProcesses.Count -eq 0) {
        Write-TimestampedMessage "Starting log monitor process..."
        Start-Process -FilePath "python" -ArgumentList "src/log_sanitizer.py --monitor" -WindowStyle Hidden
        Write-TimestampedMessage "Log monitor started."
    } else {
        Write-TimestampedMessage "Log monitor already running."
    }

} catch {
    Write-TimestampedMessage "Error running log sanitizer: $_"
    exit 1
}

# Calculate duration
$endTime = Get-Date
$duration = $endTime - $startTime
Write-TimestampedMessage "Log sanitization completed in $($duration.TotalSeconds) seconds."
