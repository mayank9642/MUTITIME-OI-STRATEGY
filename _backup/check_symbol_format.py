"""
Pre-Trading Symbol Format Check
Run this script before market open to verify symbol format is working
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import logging
from src.symbol_validator import get_recommended_format

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/symbol_validation.log'),
            logging.StreamHandler()
        ]
    )
    
    print("\n" + "="*70)
    print("FYERS SYMBOL FORMAT VALIDATION")
    print("="*70)
    print("This script checks which symbol format Fyers is currently accepting")
    print("Run this before market open or when expiry changes to ensure trading works")
    print("="*70 + "\n")
    
    result = get_recommended_format()
    
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    print(f"Format Type: {result['type'].upper()}")
    print(f"Description: {result['description']}")
    print(f"Pattern: {result['pattern']}")
    print("="*70)
    
    if result['type'] == 'full':
        print("\n✓ PASS: Strategy should work with current symbol format")
        print("  Example: NSE:NIFTY25NOV26000CE (day=25, month=NOV, strike=26000)")
    elif result['type'] == 'abbreviated':
        print("\n✓ PASS: Strategy should work with abbreviated format")
        print("  Example: NSE:NIFTY25N26000CE (day=25, month=N, strike=26000)")
        print("  Note: The retry logic will handle this automatically")
    else:
        print("\n⚠️ WARNING: Could not validate symbol format")
        print("  Possible reasons:")
        print("  - Market is closed (option chain not updating)")
        print("  - Network connectivity issues")
        print("  - Fyers API issues")
        print("\n  Recommendation:")
        print("  - Try running this again closer to market open")
        print("  - Check network connection")
        print("  - Verify Fyers API access token is valid")
    
    print("="*70 + "\n")
    
    return 0 if result['type'] in ['full', 'abbreviated'] else 1

if __name__ == "__main__":
    sys.exit(main())
