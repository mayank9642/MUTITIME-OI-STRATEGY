"""
Advanced fix for syntax issues in strategy.py
Resolves unterminated triple-quoted strings, indentation problems, and other syntax errors
"""
import os
import re
import shutil
import logging

def fix_docstring_issue():
    file_path = r'c:\vs code projects\finalized strategies\src\strategy.py'
    backup_path = file_path + '.complete_syntax_fix_' + str(int(os.path.getmtime(file_path)))
    
    print(f"Creating backup at {backup_path}")
    shutil.copy2(file_path, backup_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as src:
            content = src.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # First fix: Replace multiline docstrings with single line versions
    # This is the safest approach to deal with unterminated docstrings
    method_patterns = [
        (r'def update_trailing_stoploss\([^)]*\):\s*"""[^"]*', 
         'def update_trailing_stoploss(self, current_price):\n        """Update the trailing stoploss based on current price and profit percentage"""\n'),
        (r'def process_exit\([^)]*\):\s*"""[^"]*', 
         'def process_exit(self, exit_reason="manual", exit_price=None):\n        """Process exit consistently for all exit types (stoploss, target, time, market close)"""\n'),
        (r'def check_partial_exit\([^)]*\):\s*"""[^"]*',
         'def check_partial_exit(self):\n        """Check and execute partial exits based on predefined rules"""\n'),
        (r'def run_diagnostic\([^)]*\):\s*"""[^"]*', 
         'def run_diagnostic(self):\n        """Run a self-diagnostic check to verify key components are functioning"""\n'),
        (r'def run_strategy\([^)]*\):\s*"""[^"]*', 
         'def run_strategy(self, force_analysis=False):\n        """Main function to run the strategy"""\n'),
        (r'def save_trade_history\([^)]*\):\s*"""[^"]*', 
         'def save_trade_history(self):\n        """Save trade history to both CSV and Excel files with proper error handling"""\n'),
        (r'def record_trade_metrics\([^)]*\):\s*"""[^"]*', 
         'def record_trade_metrics(self):\n        """Record trade performance metrics for analysis and reporting"""\n'),
        (r'def update_aggregate_stats\([^)]*\):\s*"""[^"]*', 
         'def update_aggregate_stats(self):\n        """Update aggregate statistics file with new trade data"""\n'),
        (r'def get_current_time\([^)]*\):\s*"""[^"]*', 
         'def get_current_time(self):\n        """Get current time in IST timezone"""\n'),
        (r'def wait_for_market_open\([^)]*\):\s*"""[^"]*', 
         'def wait_for_market_open(self):\n        """Wait for market to open and then run the strategy"""\n'),
        (r'def quick_exit_check\([^)]*\):\s*"""[^"]*', 
         'def quick_exit_check(self):\n        """Check for immediate exit conditions (SL/target) on every monitoring loop iteration"""\n'),
        (r'def generate_daily_report\([^)]*\):\s*"""[^"]*', 
         'def generate_daily_report(self):\n        """Generate a summary report of the day\'s trading activity"""\n')
    ]
    
    new_content = content
    
    # Apply the pattern replacements
    for pattern, replacement in method_patterns:
        new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
    
    # Second fix: Directly target specific lines with known issues 
    # (e.g. unterminated strings, single quotes within double quotes, etc)
    lines = new_content.split('\n')
    fixed_lines = []
    
    # Flag to track if we're in a multiline docstring
    in_docstring = False
    
    for i, line in enumerate(lines):
        # Handle unterminated triple quotes
        if '"""' in line:
            quote_count = line.count('"""')
            if quote_count % 2 == 1:  # Odd number of triple quotes
                if not in_docstring:  # Opening quote
                    in_docstring = True
                    if not line.strip().endswith('"""'):  # If not terminated in the same line
                        line = line + '"""'
                        in_docstring = False
                else:  # Closing quote
                    in_docstring = False
        
        # Fix indentation issues - ensure consistent indentation pattern
        if re.match(r'^\s*def\s+\w+', line) and not line.strip().endswith(':'):
            line = line + ':'
        
        # Add the fixed line
        fixed_lines.append(line)
    
    # Convert back to a string
    fixed_content = '\n'.join(fixed_lines)
      # Final fix: Ensure any remaining broken docstring patterns are fixed
    fixed_content = re.sub(r'"""([^"]*?)$', r'"""\1"""', fixed_content, flags=re.MULTILINE)
    
    # Add the missing update_trailing_stoploss method if it doesn't exist
    if "def update_trailing_stoploss" not in fixed_content:
        print("Adding missing update_trailing_stoploss method")
        
        # Define the method
        trailing_sl_method = '''
    def update_trailing_stoploss(self, current_price):
        """Update the trailing stoploss based on current price and profit percentage"""
        if not self.active_trade:
            return
        
        symbol = self.active_trade.get('symbol', '')
        entry_price = self.active_trade.get('entry_price', 0)
        current_sl = self.active_trade.get('stoploss', 0)
        original_stoploss = self.active_trade.get('original_stoploss', current_sl)
        
        # First time trailing SL is called, store the original stoploss
        if 'original_stoploss' not in self.active_trade:
            self.active_trade['original_stoploss'] = current_sl
            original_stoploss = current_sl
        
        # Get trailing stop percentage from config
        config = self.config or {}
        trailing_stop_pct = config.get('strategy', {}).get('trailing_stop_pct', 8)
        
        # Calculate new potential stoploss (current price - trailing percentage)
        potential_stoploss = current_price * (1 - (trailing_stop_pct / 100))
        
        # Log debug info
        logging.info(f"TRAILING SL DEBUG | symbol: {symbol} | entry_price: {entry_price} | current_price: {current_price} | trailing_stop_pct: {trailing_stop_pct} | current_sl: {current_sl} | original_stoploss: {original_stoploss}")
        
        # For long positions, we want to move the stoploss up as price increases
        logging.info(f"TRAILING SL DEBUG | [LONG] potential_stoploss: {potential_stoploss}")
        
        # Only update if the new stoploss is higher than both current stoploss and original stoploss
        if potential_stoploss > current_sl and potential_stoploss > original_stoploss:
            old_sl = self.active_trade['stoploss']
            self.active_trade['stoploss'] = round(potential_stoploss, 3)
            self.active_trade['trailing_stoploss'] = round(potential_stoploss, 3)
            
            logging.info(f"Trailing stoploss updated from {old_sl} to {self.active_trade['stoploss']}")
            return True
        else:
            logging.info(f"TRAILING SL DEBUG | [LONG] No update: potential_stoploss ({potential_stoploss}) <= current_sl ({current_sl}) or original_stoploss ({original_stoploss})")
            return False
            
'''
        
        # Find a suitable place to insert the method (before process_exit)
        if 'def process_exit' in fixed_content:
            insert_pos = fixed_content.find('def process_exit')
            if insert_pos > 0:
                # Insert the method at the found position
                fixed_content = fixed_content[:insert_pos] + trailing_sl_method + fixed_content[insert_pos:]
                print("Successfully added update_trailing_stoploss method")
            else:
                print("Could not find a suitable insertion point for update_trailing_stoploss method")
        else:
            print("Could not find process_exit method to insert update_trailing_stoploss before it")
    
    # Write the fixed content back to the file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print("Fixed syntax issues in the file")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

if __name__ == "__main__":
    fix_docstring_issue()
