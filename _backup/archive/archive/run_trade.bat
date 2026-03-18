@echo off
echo ===============================================================
echo              RUNNING TRADING STRATEGY
echo ===============================================================

cd /d "%~dp0"
python simple_run.py > strategy_output.txt 2>&1

echo.
echo Execution completed. Check strategy_output.txt for results.
echo.
pause
