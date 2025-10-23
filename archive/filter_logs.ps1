# Filter log files to remove sensitive information
Write-Host "Filtering log files to remove sensitive information..." -ForegroundColor Green
$scriptPath = $PSScriptRoot
Set-Location $scriptPath
python src/filter_logs.py
Write-Host "Log files filtered successfully." -ForegroundColor Green
