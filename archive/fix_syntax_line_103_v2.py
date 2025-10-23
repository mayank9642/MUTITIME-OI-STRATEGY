"""
Super focused fix for line 103 syntax error
"""

def fix_line_103():
    file_path = 'c:\\vs code projects\\finalized strategies\\src\\strategy.py'
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split into individual lines for precise editing
    lines = content.split('\n')
    
    # Look for the problematic line
    for i in range(len(lines)):
        # Exact match for the problematic line
        if '"""Reset all state variables for a clean start"""        # OI analysis results"""' in lines[i]:
            print(f"Found problematic line at line {i+1}")
            # Replace with fixed version
            lines[i] = '        """Reset all state variables for a clean start"""'
            lines.insert(i+1, '        # OI analysis results')
            print("Fixed line by splitting into two lines")
            break
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("File updated successfully")

if __name__ == "__main__":
    fix_line_103()
