# PowerShell script to run the strategy with market open waiting
Write-Host "Running Options Trading Strategy with Market Open Waiting..." -ForegroundColor Green
Write-Host "This script will wait for market to open before executing the strategy" -ForegroundColor Cyan
python wait_and_run_strategy.py
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
