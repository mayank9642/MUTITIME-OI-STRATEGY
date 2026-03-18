"""
Fix for specific syntax error at line 103 in strategy.py
"""
import os

def fix_line_103():
    strategy_file = 'c:\\vs code projects\\finalized strategies\\src\\strategy.py'
    
    # Read the file
    with open(strategy_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # Find the problematic line
    for i, line in enumerate(lines):
        if '"""Reset all state variables for a clean start"""' in line and '# OI analysis results"""' in line:
            lines[i] = '        """Reset all state variables for a clean start"""\n        # OI analysis results\n'
            print(f"Fixed problematic line {i+1}")
    
    # Write the file back
    with open(strategy_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("File updated successfully")

if __name__ == "__main__":
    fix_line_103()
