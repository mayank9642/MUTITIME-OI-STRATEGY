"""
Debug syntax errors in strategy.py

This script analyzes the Python syntax in strategy.py and finds syntax errors
"""
import tokenize
import io
import re
import os

def debug_file():
    strategy_file = 'c:\\vs code projects\\finalized strategies\\src\\strategy.py'
    
    # Read the file
    with open(strategy_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split into lines for analysis
    lines = content.split('\n')
    
    # Search for problematic docstrings (unterminated or incorrectly terminated)
    print("Checking for problematic docstrings...")
    
    # Look for triple quotes that might be problematic
    line_num = 0
    in_docstring = False
    docstring_start = -1
    triple_quotes = []
    
    for i, line in enumerate(lines):
        # Count triple quotes in this line
        matches = re.finditer(r'"""', line)
        positions = [match.start() for match in matches]
        
        for pos in positions:
            triple_quotes.append((i, pos))
            
    # If we have an odd number of triple quotes, that's a problem
    if len(triple_quotes) % 2 != 0:
        print(f"CRITICAL: Odd number of triple quotes detected: {len(triple_quotes)}")
        
    # Check for lines with odd number of triple quotes
    for i, line in enumerate(lines):
        count = line.count('"""')
        if count % 2 != 0:
            print(f"Line {i+1} has odd number of triple quotes: {count}")
            print(f"  Content: {line}")
    
    # Check for specific syntax around line 103
    target_line = 102  # 0-indexed
    context_lines = 10
    
    print(f"\nAnalyzing context around line {target_line+1}:")
    for i in range(max(0, target_line - context_lines), min(len(lines), target_line + context_lines + 1)):
        if i == target_line:
            print(f">>> {i+1}: {lines[i]}")
        else:
            print(f"    {i+1}: {lines[i]}")
    
    # Write a clean version of the problem section
    print("\nAttempting to fix the reset_state method...")
    
    # Find the start of the reset_state method
    reset_state_start = -1
    reset_state_end = -1
    
    for i, line in enumerate(lines):
        if "def reset_state" in line:
            reset_state_start = i
            break
    
    # Find the end of reset_state method
    if reset_state_start >= 0:
        for i in range(reset_state_start + 1, len(lines)):
            if re.match(r'\s*def\s+', lines[i]):
                reset_state_end = i - 1
                break
                
        if reset_state_end == -1:  # If no next method, go to end of file
            reset_state_end = len(lines) - 1
            
        print(f"reset_state method spans lines {reset_state_start+1} to {reset_state_end+1}")
        
        # Create a fixed version
        fixed_reset_state = [
            "    def reset_state(self):",
            '        """Reset all state variables for a clean start"""',
            "        # OI analysis results",
        ]
        
        # Add the rest of the method body
        in_body = False
        for i in range(reset_state_start + 1, reset_state_end + 1):
            line = lines[i]
            if line.strip().startswith("self.") or in_body:
                in_body = True
                fixed_reset_state.append(line)
        
        print("\nFixed reset_state method:")
        for line in fixed_reset_state:
            print(line)
            
        # Write fixed reset_state back to the file
        lines[reset_state_start:reset_state_end+1] = fixed_reset_state
        
        with open(strategy_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            
        print("\nFile updated with fixed reset_state method")
        
    # Now check for trailing_stoploss method
    print("\nChecking for update_trailing_stoploss method...")
    
    has_trailing_stoploss = False
    for i, line in enumerate(lines):
        if "def update_trailing_stoploss" in line:
            has_trailing_stoploss = True
            print(f"Found update_trailing_stoploss at line {i+1}")
            
    if not has_trailing_stoploss:
        print("WARNING: update_trailing_stoploss method not found!")
        
if __name__ == "__main__":
    debug_file()
