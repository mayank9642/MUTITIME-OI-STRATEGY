"""
Improved patch script for fixing fyers_api_utils.py
"""
import sys
import os
import re

def fix_indentation():
    file_path = os.path.join('src', 'fyers_api_utils.py')
    
    # Create backup
    backup_path = file_path + '.bak'
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    print(f"Created backup at {backup_path}")
    
    # Fix specific indentation issues using line-by-line approach
    lines = original_content.split('\n')
    
    for i in range(len(lines)):
        # Fix specific issues we identified
        if re.match(r'\s+def on_subscribe_success', lines[i]) and '  def on_subscribe_success' in lines[i]:
            lines[i] = '    def on_subscribe_success' + lines[i].split('def on_subscribe_success')[1]
            print(f"Fixed line {i+1}: on_subscribe_success function definition")
            
        if re.match(r'\s+def on_subscribe_failure', lines[i]) and '  def on_subscribe_failure' in lines[i]:
            lines[i] = '    def on_subscribe_failure' + lines[i].split('def on_subscribe_failure')[1]
            print(f"Fixed line {i+1}: on_subscribe_failure function definition")
            
        # Fix the missing newline issue
        if 'logged_symbols.clear()                market_data_df.last_symbol_clear = now' in lines[i]:
            lines[i] = '                logged_symbols.clear()'
            lines.insert(i+1, '                market_data_df.last_symbol_clear = now')
            print(f"Fixed line {i+1}: inserted missing newline")
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Fixed file written to {file_path}")

if __name__ == "__main__":
    fix_indentation()
