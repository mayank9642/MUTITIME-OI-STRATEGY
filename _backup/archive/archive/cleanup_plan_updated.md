# Project Cleanup Plan (Updated)

This document outlines the files to keep and remove for the finalized strategies project.

## Core Files to Keep

### Main Strategy and API Files
- `src/strategy.py` - Core strategy implementation
- `src/fixed_strategy.py` - Fixed version of the strategy
- `src/fyers_api_utils.py` - Fyers API utilities
- `src/nse_data_new.py` - NSE data fetching utilities
- `src/symbol_formatter.py` - Option symbol formatting utilities
- `src/improved_websocket.py` - Enhanced WebSocket implementation
- `src/main.py` - Main entry point for scheduled running
- `src/token_helper.py` - Token management utilities
- `src/auth.py` - Authentication utilities
- `src/config.py` - Configuration management

### Main Run Scripts
- `run_strategy.py` - Main script to run the strategy with fixes
- `run_fixed_strategy.py` - Alternative main script (if needed)
- `wait_and_run_strategy.py` - Script for waiting for market open

### Configuration and Documentation
- `config/` - Directory with configuration files
- `README.md` - Main documentation
- `requirements.txt` - Dependencies
- `DIRECTORY_STRUCTURE.md` - Directory structure documentation
- `docs/` - Documentation directory
- `cleanup_report.md` - Report from cleanup process (will be generated)

## Files to Remove

### Test Files
- All files with "test" in the name that aren't core test utilities
- Files for checking implementation details:
  - `simple_strategy_test.py`
  - `test_trailing_stop_loss.py`
  - `direct_test.py`
  - `simple_test.py`
  - `check_methods.py`
  - `method_check.py`
  - `verify_methods.py`
  - `simple_check.py`
  - `ultimate_simple_test.py`
  - `final_test.py`
  - `bare_minimum_test.py`
  - `test_symbol_conversion.py`
  - `test_fyers_option_format.py`

### Duplicate Run Scripts
- `run_simple_strategy.py`
- `run_standalone.py`
- `final_run.py`
- `run_fixed.py`
- `run_strategy_with_fixed_format.py`
- `run_strategy_with_improved_websocket.py`

### Fix and Debugging Files
- All files starting with `fix_`
- All files starting with `debug_`
- Fix-related files:
  - `direct_fix_trailing_stoploss.py`
  - `comprehensive_syntax_fix.py`
  - `enhanced_syntax_fix.py`
  - `direct_comprehensive_fix.py`
  - `patch_strategy.py`
  - `standalone_strategy.py`
  - `standalone_trailing_test.py`
  - `fix_strategy_clean.py`
  - `fix_all_issues.py`
  - `fix_trailing_stoploss.py`

### Batch Files
- `fix_and_run.bat`
- `run_tests_and_strategy.bat`
- `run_strategy_filtered.bat`
- `run_strategy_with_improved_websocket.bat`
- `run_strategy_secure.bat`

### Backup and Temporary Files
- All files with `.bak` extension
- All files with `.old` extension
- All files with "backup" in the name
- Strategy backup files (like `strategy.py.backup_*`)
- Any other backup files in the `src` directory

### Archive Directory
- The entire `archive` directory will be backed up and removed

## Rationale
- Keep only core files needed for strategy execution
- Remove all test, debug, and fix files that were created during development
- Maintain a clean and focused directory structure
- Keep documentation and configuration files intact

## Implementation
The cleanup process is implemented in `cleanup_project.ps1` (PowerShell script) and can be run using `run_cleanup.bat`. The process:

1. Creates backups of all files to be removed
2. Removes test files
3. Removes duplicate run scripts
4. Removes fix and debugging files
5. Removes unnecessary batch files
6. Removes files with specific prefixes (fix_, debug_)
7. Removes backup and temporary files
8. Processes and removes the archive directory
9. Cleans up src directory backup files
10. Generates a detailed cleanup report
