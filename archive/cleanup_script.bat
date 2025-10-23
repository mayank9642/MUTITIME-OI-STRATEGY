@echo off
REM Batch script to run the PowerShell cleanup script

echo ==========================================================
echo          Running cleanup script for strategy project     
echo ==========================================================

powershell.exe -ExecutionPolicy Bypass -File "%~dp0cleanup_script.ps1"

echo.
echo ==========================================================
echo                  Cleanup completed                       
echo ==========================================================

pause
