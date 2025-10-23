@echo off
:: sanitize_logs.bat
:: Script to sanitize log files by removing sensitive information
:: Run this script periodically or as a scheduled task

echo %date% %time% - Starting log sanitization process...

:: Change to script directory
cd /d "%~dp0"

:: Run the log sanitizer
echo %date% %time% - Running log sanitizer script...
python src/log_sanitizer.py
if %ERRORLEVEL% neq 0 (
    echo %date% %time% - Error running log sanitizer. Exit code: %ERRORLEVEL%
    exit /b 1
)

echo %date% %time% - Log sanitization completed successfully.
