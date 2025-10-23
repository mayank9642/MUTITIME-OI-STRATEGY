@echo off
echo Running strategy with filtered logging...
cd /d "%~dp0"
python src/filter_logs.py
echo Log file has been filtered to remove sensitive information.
