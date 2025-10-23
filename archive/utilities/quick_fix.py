"""
Quick fix for the fyers_api_utils.py file to run the strategy
"""
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Read the original file
file_path = 'src/fyers_api_utils.py'
with open(file_path, 'r') as f:
    content = f.read()

# Fix indentation issues
content = content.replace("          def on_subscribe_success", "    def on_subscribe_success")
content = content.replace("              def on_subscribe_failure", "    def on_subscribe_failure")
content = content.replace("                logged_symbols.clear()", "                logged_symbols.clear()\n")

# Create a backup of the original file
backup_path = 'src/fyers_api_utils.py.bak'
with open(backup_path, 'w') as f:
    f.write(content)

# Write the fixed content
with open(file_path, 'w') as f:
    f.write(content)

print(f"Fixed indentation issues in {file_path}")
print(f"Original file backed up to {backup_path}")
