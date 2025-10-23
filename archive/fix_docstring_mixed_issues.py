"""
Enhanced docstring and comment fix for strategy.py

This script specifically fixes:
1. Mixed docstrings and comments causing syntax errors
2. Duplicate methods
3. Missing method implementations
"""
import re
import os
import shutil
from datetime import datetime

def fix_docstring_issues():
    strategy_file = 'c:\\vs code projects\\finalized strategies\\src\\strategy.py'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{strategy_file}.backup_{timestamp}"
    
    # Create backup with timestamp
    print(f"Creating backup of {strategy_file} to {backup_file}")
    shutil.copy2(strategy_file, backup_file)
    
    try:
        with open(strategy_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
        
    # 1. Fix mixed docstrings and comments (most critical issue)
    # Example: """Reset all state variables for a clean start"""        # OI analysis results"""
    content = re.sub(r'"""([^"]*)"""(\s*#[^"]*?)"""', r'"""\1"""', content)
    content = re.sub(r'"""([^"]*)"""(\s*#[^"]*)', r'"""\1"""\2', content)
    
    # 2. Fix specific problematic method docstrings
    problematic_methods = {
        'reset_state': r'def reset_state\(self\):[^"]*?"""[^"]*?""".*?#.*?"""',
        'initialize_day': r'def initialize_day\(self\):[^"]*?"""[^"]*?""".*?#.*?"""',
        'identify_high_oi_strikes': r'def identify_high_oi_strikes\(self\):[^"]*?"""[^"]*?""".*?#.*?"""',
        '_find_suitable_strikes': r'def _find_suitable_strikes\([^)]*\):[^"]*?"""[^"]*?""".*?#.*?"""'
    }
    
    for method, pattern in problematic_methods.items():
        if re.search(pattern, content, re.DOTALL):
            fixed = re.sub(pattern, lambda m: m.group(0).split('"""', 2)[0] + '"""' + m.group(0).split('"""', 2)[1] + '"""', content, flags=re.DOTALL)
            if fixed != content:
                content = fixed
                print(f"Fixed problematic docstring in method: {method}")
    
    # 3. Fix reset_state specifically (line 103)
    reset_state_fix = '''def reset_state(self):
        """Reset all state variables for a clean start"""
        # OI analysis results
        self.highest_put_oi_strike = None
        self.highest_call_oi_strike = None'''
        
    content = re.sub(r'def reset_state\(self\):.*?self\.highest_call_oi_strike = None', reset_state_fix, content, flags=re.DOTALL)
    
    # 4. Fix all unterminated docstrings
    content = re.sub(r'"""([^"]*)$', r'"""\1"""', content, flags=re.MULTILINE)
    
    # 5. Fix excessive quotes in docstring terminations
    content = re.sub(r'"{6}', r'"""', content)
    
    # 6. Ensure update_trailing_stoploss method exists
    if 'def update_trailing_stoploss' not in content:
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

        # Find a good position to insert it (before process_exit if it exists, otherwise at the end of the class)
        process_exit_pos = content.find('def process_exit')
        if process_exit_pos > 0:
            content = content[:process_exit_pos] + trailing_sl_method + content[process_exit_pos:]
            print("Added missing update_trailing_stoploss method")
        else:
            # Try to find the end of the class to insert there
            class_end = content.rfind('\n\n')
            if class_end > 0:
                content = content[:class_end] + trailing_sl_method + content[class_end:]
                print("Added missing update_trailing_stoploss method at end of class")
    else:
        print("update_trailing_stoploss method already exists")
    
    # 7. Find duplicate method definitions and remove them
    method_pattern = r'def\s+(\w+)\s*\('
    method_matches = list(re.finditer(method_pattern, content))
    method_names = [match.group(1) for match in method_matches]
    
    # Identify duplicates
    duplicate_methods = {}
    for i, name in enumerate(method_names):
        if name in duplicate_methods:
            duplicate_methods[name].append(i)
        else:
            duplicate_methods[name] = [i]
    
    # Keep only methods with duplicates
    duplicate_methods = {name: indices for name, indices in duplicate_methods.items() if len(indices) > 1}
    
    if duplicate_methods:
        print(f"Found duplicate methods: {list(duplicate_methods.keys())}")
        
        # Split content into lines for easier processing
        lines = content.split('\n')
        lines_to_remove = set()
        
        for method_name, indices in duplicate_methods.items():
            # Skip the first occurrence (keep it)
            for duplicate_idx in indices[1:]:
                match = method_matches[duplicate_idx]
                method_start_line = content[:match.start()].count('\n')
                
                # Find the end of this method (next method or end of file)
                next_method_idx = duplicate_idx + 1
                method_end_line = len(lines)
                
                if next_method_idx < len(method_matches):
                    next_method_start = method_matches[next_method_idx].start()
                    method_end_line = content[:next_method_start].count('\n')
                
                # Mark lines to remove
                for i in range(method_start_line, method_end_line):
                    lines_to_remove.add(i)
                
                print(f"Marked duplicate method '{method_name}' (lines {method_start_line}-{method_end_line}) for removal")
        
        # Remove marked lines
        if lines_to_remove:
            new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
            content = '\n'.join(new_lines)
            print(f"Removed {len(lines_to_remove)} lines containing duplicate methods")
    
    # Write fixed content back to file
    try:
        with open(strategy_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully fixed docstring issues in {strategy_file}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

if __name__ == "__main__":
    success = fix_docstring_issues()
    if success:
        print("✓ Successfully fixed docstring issues in strategy.py")
    else:
        print("✗ Failed to fix docstring issues in strategy.py")
