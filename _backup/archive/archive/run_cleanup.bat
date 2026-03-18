@echo off
echo ===============================================================
echo              FINALIZED STRATEGIES CLEANUP UTILITY
echo ===============================================================
echo This will clean up the project by removing test files, duplicates,
echo backups, and other unnecessary files. All removed files will be
echo backed up first.
echo.
echo Press any key to continue or CTRL+C to cancel...
pause > nul

powershell -ExecutionPolicy Bypass -File "cleanup_project.ps1"

echo.
echo Cleanup complete! Check cleanup_report.md for details.
echo.
pause
