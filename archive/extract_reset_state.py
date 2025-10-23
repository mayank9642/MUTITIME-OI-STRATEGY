"""
Extract reset_state from strategy_fixed.py and replace it in strategy.py
"""
import re

def extract_and_replace_reset_state():
    fixed_file = 'c:\\vs code projects\\finalized strategies\\src\\strategy_fixed.py'
    target_file = 'c:\\vs code projects\\finalized strategies\\src\\strategy.py'
    
    # Read the fixed file
    with open(fixed_file, 'r', encoding='utf-8', errors='replace') as f:
        fixed_content = f.read()
    
    # Extract the reset_state method from fixed file
    reset_pattern = r'def reset_state\(self\):(.*?)(?=\n    def|\Z)'
    fixed_match = re.search(reset_pattern, fixed_content, re.DOTALL)
    
    if not fixed_match:
        print("Could not find reset_state method in fixed file")
        return False
    
    fixed_reset = 'def reset_state(self):' + fixed_match.group(1)
    print("Extracted reset_state method from fixed file")
    
    # Read the target file
    with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
        target_content = f.read()
    
    # Find the reset_state method in target file
    target_match = re.search(reset_pattern, target_content, re.DOTALL)
    
    if not target_match:
        print("Could not find reset_state method in target file")
        return False
    
    # Replace it
    target_reset = 'def reset_state(self):' + target_match.group(1)
    new_content = target_content.replace(target_reset, fixed_reset)
    
    # Write back
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully replaced reset_state method")
    return True

if __name__ == "__main__":
    extract_and_replace_reset_state()
