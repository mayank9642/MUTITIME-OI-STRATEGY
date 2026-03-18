# PowerShell script to perform routine cleanup operations
# Cleans Python cache files and organizes logs

# Define the project root directory
$projectRoot = $PSScriptRoot

Write-Host "=========================================================" -ForegroundColor Green
Write-Host "          Running cleanup script for strategy project     " -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green

# 1. Remove Python cache files
Write-Host "Removing Python cache files..." -ForegroundColor Yellow
Get-ChildItem -Path $projectRoot -Filter "__pycache__" -Recurse | ForEach-Object {
    Remove-Item -Path $_.FullName -Recurse -Force
    Write-Host "Removed: $($_.FullName)" -ForegroundColor Gray
}

# 2. Move old log files to backups
Write-Host "Moving old log files to backups..." -ForegroundColor Yellow

# Create backups logs directory if it doesn't exist
if (-not (Test-Path "$projectRoot\backups\logs")) {
    New-Item -Path "$projectRoot\backups\logs" -ItemType Directory -Force | Out-Null
}

# Keep the 5 most recent log backups, move older ones
Get-ChildItem -Path "$projectRoot\logs" -Filter "*.log.bak" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -Skip 5 | 
    ForEach-Object {
        Move-Item -Path $_.FullName -Destination "$projectRoot\backups\logs\$($_.Name)" -Force
        Write-Host "Moved old log: $($_.Name)" -ForegroundColor Gray
    }

# 3. Backup current log files
Write-Host "Backing up current log files..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Get-ChildItem -Path "$projectRoot\logs" -Filter "*.log" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination "$projectRoot\backups\logs\$($_.BaseName)_$timestamp.log.bak" -Force
    Write-Host "Backed up: $($_.Name)" -ForegroundColor Gray
}

Write-Host "=========================================================" -ForegroundColor Green
Write-Host "                  Cleanup completed                       " -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green

Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
