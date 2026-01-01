"""
PERMANENT SOLUTION: Adaptive Symbol Format Manager
Auto-detects and persists working Fyers symbol formats to eliminate monthly breakage.

This solves the recurring problem where Fyers changes symbol formats every month,
breaking our strategies. Instead of manual fixes, this system:

1. Tests ALL symbol patterns found in option chain
2. Persists the working pattern to a cache file  
3. Auto-updates when formats change
4. Falls back gracefully when current format breaks
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.fetch_option_oi import fetch_option_oi
try:
    from src.fixed_improved_websocket import enhanced_start_market_data_websocket
except ImportError:
    from fixed_improved_websocket import enhanced_start_market_data_websocket

SYMBOL_CACHE_FILE = "config/working_symbol_format.json"

class AdaptiveSymbolManager:
    """
    Automatically learns and adapts to Fyers symbol format changes.
    Eliminates the need for monthly manual fixes.
    """
    
    def __init__(self):
        self.working_format = None
        self.format_cache = {}
        self.load_cached_format()
    
    def load_cached_format(self):
        """Load previously working format from cache"""
        try:
            if os.path.exists(SYMBOL_CACHE_FILE):
                with open(SYMBOL_CACHE_FILE, 'r') as f:
                    self.format_cache = json.load(f)
                    
                # Check if cached format is recent (within 7 days)
                if 'last_validated' in self.format_cache:
                    last_validated = datetime.fromisoformat(self.format_cache['last_validated'])
                    if datetime.now() - last_validated < timedelta(days=7):
                        self.working_format = self.format_cache
                        logging.info(f"✓ Using cached working format: {self.working_format.get('pattern_name', 'unknown')}")
                        return
                        
                logging.info("Cached format is stale, will re-validate")
        except Exception as e:
            logging.warning(f"Could not load cached format: {e}")
    
    def save_working_format(self, format_info: Dict):
        """Persist working format to cache"""
        try:
            os.makedirs(os.path.dirname(SYMBOL_CACHE_FILE), exist_ok=True)
            format_info['last_validated'] = datetime.now().isoformat()
            
            with open(SYMBOL_CACHE_FILE, 'w') as f:
                json.dump(format_info, f, indent=2)
                
            logging.info(f"✓ Saved working format to cache: {format_info.get('pattern_name')}")
        except Exception as e:
            logging.error(f"Failed to save format cache: {e}")
    
    def extract_symbol_patterns(self, oi_data) -> List[Dict]:
        """
        Extract all unique symbol patterns from option chain data.
        Returns list of pattern info dicts for testing.
        """
        if oi_data is None or len(oi_data) == 0:
            return []
        
        patterns = []
        seen_patterns = set()
        
        # Get sample symbols for pattern detection
        sample_symbols = oi_data['symbol'].dropna().unique()[:50]  # Test up to 50 patterns
        
        for symbol in sample_symbols:
            pattern_info = self.analyze_symbol_pattern(symbol)
            if pattern_info and pattern_info['pattern_signature'] not in seen_patterns:
                patterns.append(pattern_info)
                seen_patterns.add(pattern_info['pattern_signature'])
        
        # Sort by preference (D-code patterns first, then others)
        patterns.sort(key=lambda x: (
            0 if 'D' in x['pattern_name'] else 1,  # D-code patterns first
            1 if 'JAN' in x['example'] or 'Jan' in x['example'] else 0,  # January patterns last
            x['pattern_name']
        ))
        
        return patterns
    
    def analyze_symbol_pattern(self, symbol: str) -> Optional[Dict]:
        """Analyze a symbol to extract its pattern info"""
        symbol = symbol.strip()
        
        # Pattern 1: D-code format (NSE:NIFTY25D0923750CE)
        match = re.match(r"NSE:([A-Z]+)(\d{2})D(\d{2})(\d{5,6})(CE|PE)", symbol, re.IGNORECASE)
        if match:
            return {
                'pattern_name': 'dcode',
                'pattern_signature': 'NSE_UNDERLYING_YYD_DD_STRIKE_CYPE',
                'regex': r"NSE:([A-Z]+)(\d{2})D(\d{2})(\d{5,6})(CE|PE)",
                'example': symbol,
                'description': 'D-code weekly format (YYD format)',
                'components': {
                    'underlying': match.group(1),
                    'year': match.group(2),
                    'day_code': match.group(3),
                    'strike': match.group(4),
                    'option_type': match.group(5)
                }
            }
        
        # Pattern 2: Monthly January format (NSE:NIFTY06JAN24250PE)
        match = re.match(r"NSE:([A-Z]+)(\d{2})([A-Z]{3})(\d{5,6})(CE|PE)", symbol, re.IGNORECASE)
        if match:
            return {
                'pattern_name': 'monthly_abbreviated',
                'pattern_signature': 'NSE_UNDERLYING_DD_MMM_STRIKE_CYPE',
                'regex': r"NSE:([A-Z]+)(\d{2})([A-Z]{3})(\d{5,6})(CE|PE)",
                'example': symbol,
                'description': 'Monthly abbreviated format (DDMon)',
                'components': {
                    'underlying': match.group(1),
                    'day': match.group(2),
                    'month': match.group(3),
                    'strike': match.group(4),
                    'option_type': match.group(5)
                }
            }
        
        # Pattern 3: Full date format (NSE:NIFTY-25-Dec-2025-26000-CE)
        match = re.match(r"NSE:([A-Z]+)-(\d{2})-([A-Za-z]{3})-(\d{4})-(\d{5,6})-(CE|PE)", symbol, re.IGNORECASE)
        if match:
            return {
                'pattern_name': 'full_date',
                'pattern_signature': 'NSE_UNDERLYING_DD_MON_YYYY_STRIKE_CYPE',
                'regex': r"NSE:([A-Z]+)-(\d{2})-([A-Za-z]{3})-(\d{4})-(\d{5,6})-(CE|PE)",
                'example': symbol,
                'description': 'Full date format with dashes',
                'components': {
                    'underlying': match.group(1),
                    'day': match.group(2),
                    'month': match.group(3),
                    'year': match.group(4),
                    'strike': match.group(5),
                    'option_type': match.group(6)
                }
            }
        
        return None
    
    def test_symbol_pattern(self, pattern_info: Dict, test_symbols: List[str]) -> bool:
        """
        Test if a pattern works by attempting WebSocket subscription.
        Returns True if pattern is working, False otherwise.
        """
        if not test_symbols:
            return False
            
        try:
            logging.info(f"Testing pattern '{pattern_info['pattern_name']}' with symbols: {test_symbols[:2]}")
            
            # Attempt WebSocket connection with these symbols
            ws = enhanced_start_market_data_websocket(test_symbols[:3], lambda *args: None)
            
            if ws is None:
                logging.warning(f"WebSocket connection failed for pattern {pattern_info['pattern_name']}")
                return False
            
            # Test subscription - wait briefly for response
            import time
            time.sleep(2)
            
            # If we reach here without exceptions, pattern likely works
            ws.disconnect() if hasattr(ws, 'disconnect') else None
            logging.info(f"✓ Pattern '{pattern_info['pattern_name']}' appears to be working")
            return True
            
        except Exception as e:
            logging.warning(f"Pattern '{pattern_info['pattern_name']}' test failed: {str(e)}")
            return False
    
    def find_working_symbol_format(self) -> Optional[Dict]:
        """
        Main method: Find current working symbol format by testing all patterns.
        Returns working format info or None if all fail.
        """
        # Use cached format if available and recent
        if self.working_format:
            logging.info("Using cached working format (validated within 7 days)")
            return self.working_format
        
        logging.info("🔍 AUTO-DETECTING WORKING SYMBOL FORMAT")
        
        # Fetch current option chain
        oi_data = fetch_option_oi('NIFTY')
        if oi_data is None or len(oi_data) == 0:
            logging.error("Cannot detect format - no option chain data available")
            return None
        
        # Extract all unique patterns from option chain
        patterns = self.extract_symbol_patterns(oi_data)
        logging.info(f"Found {len(patterns)} unique symbol patterns to test")
        
        # Test each pattern to find working one
        for i, pattern_info in enumerate(patterns, 1):
            logging.info(f"Testing pattern {i}/{len(patterns)}: {pattern_info['pattern_name']}")
            logging.info(f"Example: {pattern_info['example']}")
            
            # Get test symbols matching this pattern
            test_symbols = self.get_symbols_for_pattern(oi_data, pattern_info)
            
            if self.test_symbol_pattern(pattern_info, test_symbols):
                # Found working pattern!
                working_format = {
                    **pattern_info,
                    'test_symbols': test_symbols[:5],  # Save some test symbols
                    'validation_date': datetime.now().isoformat()
                }
                
                self.working_format = working_format
                self.save_working_format(working_format)
                
                logging.info(f"🎯 FOUND WORKING FORMAT: {pattern_info['pattern_name']}")
                logging.info(f"Example symbol: {pattern_info['example']}")
                return working_format
        
        logging.error("❌ NO WORKING SYMBOL FORMAT FOUND - All patterns failed!")
        return None
    
    def get_symbols_for_pattern(self, oi_data, pattern_info: Dict) -> List[str]:
        """Get symbols matching a specific pattern from option chain"""
        try:
            # Use regex to filter symbols matching this pattern
            pattern_regex = pattern_info['regex']
            matching_symbols = oi_data[
                oi_data['symbol'].str.match(pattern_regex, case=False, na=False)
            ]['symbol'].head(10).tolist()
            
            return matching_symbols
        except Exception as e:
            logging.warning(f"Error filtering symbols for pattern: {e}")
            return []
    
    def get_current_working_format(self) -> Optional[Dict]:
        """Get current working format (cached or auto-detected)"""
        if not self.working_format:
            self.working_format = self.find_working_symbol_format()
        return self.working_format

# Global instance for easy access
adaptive_symbol_manager = AdaptiveSymbolManager()

def get_working_symbol_format() -> Optional[Dict]:
    """
    Main API function: Get current working symbol format.
    Auto-detects if needed, uses cache if available.
    """
    return adaptive_symbol_manager.get_current_working_format()

def validate_symbol_format_adaptive() -> Optional[str]:
    """
    Drop-in replacement for validate_symbol_format() that uses adaptive detection.
    Returns format type string for backward compatibility.
    """
    working_format = get_working_symbol_format()
    
    if working_format:
        return working_format.get('pattern_name', 'unknown')
    else:
        return None