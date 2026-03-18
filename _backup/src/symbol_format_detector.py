"""
Automatic symbol format detector - adapts to Fyers format changes
Learns the correct format by testing live samples from Fyers API
"""
import logging
import re
from typing import Optional, Dict, Tuple
from datetime import datetime

def detect_fyers_symbol_format(sample_oi_data) -> Dict[str, str]:
    """
    Analyze sample option symbols from Fyers to detect the current format.
    
    Args:
        sample_oi_data: DataFrame with option chain data including 'symbol' column
        
    Returns:
        Dict with format details:
        {
            'pattern': regex pattern that matches the format,
            'example': example symbol,
            'format_type': 'full_month' or 'abbreviated_month' or 'unknown',
            'expiry_pattern': extracted expiry pattern (e.g., '25NOV' or '25N25'),
            'description': human-readable description
        }
    """
    if sample_oi_data is None or len(sample_oi_data) == 0:
        return {
            'pattern': None,
            'example': None,
            'format_type': 'unknown',
            'expiry_pattern': None,
            'description': 'No data available to detect format'
        }
    
    # Get a sample CE symbol
    ce_symbols = sample_oi_data[sample_oi_data['symbol'].str.contains('CE', na=False)]['symbol'].head(5)
    
    if len(ce_symbols) == 0:
        return {
            'pattern': None,
            'example': None,
            'format_type': 'unknown',
            'expiry_pattern': None,
            'description': 'No CE symbols found in data'
        }
    
    sample_symbol = ce_symbols.iloc[0]
    
    # Try to detect the format by analyzing the symbol structure
    # Expected patterns:
    # 1. NSE:NIFTY25NOV26000CE (full month - 3 letters)
    # 2. NSE:NIFTY25N2526000CE (abbreviated - 1 letter + year)
    # 3. NSE:NIFTY-25-Nov-2025-26000-CE (dash-separated format)
    # 4. nse:nifty25d0226300ce (lowercase, December/weekly style with D/d code)
    
    logging.info(f"Detecting format from sample symbol: {sample_symbol}")
    
    # Pattern 1: Full month format (25NOV26000CE)
    match = re.match(r"(?i)NSE:([A-Z]+)(\d{2})([A-Z]{3})(\d{5})(CE|PE)", sample_symbol)
    if match:
        underlying = match.group(1)
        day = match.group(2)
        month = match.group(3)
        strike = match.group(4)
        opt_type = match.group(5)
        
        return {
            'pattern': r"NSE:([A-Z]+)(\d{2})([A-Z]{3})(\d{5})(CE|PE)",
            'example': sample_symbol,
            'format_type': 'full_month',
            'expiry_pattern': f'{day}{month}',
            'description': f'Full month format: {underlying}{day}{month}{strike}{opt_type}',
            'components': {
                'underlying': underlying,
                'day': day,
                'month': month,
                'strike': strike,
                'option_type': opt_type
            }
        }
    
    # Pattern 2: Abbreviated format with year (25N2526000CE)
    match = re.match(r"(?i)NSE:([A-Z]+)(\d{2})([A-Z])(\d{2})(\d{5})(CE|PE)", sample_symbol)
    if match:
        underlying = match.group(1)
        day = match.group(2)
        month_abbr = match.group(3)
        year = match.group(4)
        strike = match.group(5)
        opt_type = match.group(6)
        
        return {
            'pattern': r"NSE:([A-Z]+)(\d{2})([A-Z])(\d{2})(\d{5})(CE|PE)",
            'example': sample_symbol,
            'format_type': 'abbreviated_month',
            'expiry_pattern': f'{day}{month_abbr}{year}',
            'description': f'Abbreviated format: {underlying}{day}{month_abbr}{year}{strike}{opt_type}',
            'components': {
                'underlying': underlying,
                'day': day,
                'month': month_abbr,
                'year': year,
                'strike': strike,
                'option_type': opt_type
            }
        }
    
    # Pattern 3: Dash-separated format (NSE:NIFTY-25-Nov-2025-26000-CE)
    match = re.match(r"(?i)NSE:([A-Z]+)-(\d{2})-([A-Za-z]{3})-(\d{4})-(\d{5})-(CE|PE)", sample_symbol)
    if match:
        underlying = match.group(1)
        day = match.group(2)
        month = match.group(3)
        year = match.group(4)
        strike = match.group(5)
        opt_type = match.group(6)
        
        return {
            'pattern': r"(?i)NSE:([A-Z]+)-(\d{2})-([A-Za-z]{3})-(\d{4})-(\d{5})-(CE|PE)",
            'example': sample_symbol,
            'format_type': 'dash_separated',
            'expiry_pattern': f'{day}-{month}-{year}',
            'description': f'Dash-separated format: {underlying}-{day}-{month}-{year}-{strike}-{opt_type}',
            'components': {
                'underlying': underlying,
                'day': day,
                'month': month,
                'year': year,
                'strike': strike,
                'option_type': opt_type
            }
        }

    # Pattern 4: December/weekly style with D/d code (e.g., nse:nifty25d0226300ce)
    match = re.match(r"(?i)NSE:([A-Z]+)(\d{2})[dD](\d{2})(\d{5,6})(CE|PE)", sample_symbol)
    if match:
        underlying = match.group(1)
        year = match.group(2)
        dcode = match.group(3)
        strike = match.group(4)
        opt_type = match.group(5)

        return {
            'pattern': r"(?i)NSE:([A-Z]+)(\d{2})[dD](\d{2})(\d{5,6})(CE|PE)",
            'example': sample_symbol,
            'format_type': 'dec_dcode',
            'expiry_pattern': f'{year}D{dcode}',
            'description': f'December/weekly D-code format: {underlying}{year}D{dcode}{strike}{opt_type}',
            'components': {
                'underlying': underlying,
                'year': year,
                'dcode': dcode,
                'strike': strike,
                'option_type': opt_type
            }
        }
    
    # Unknown format - log for manual review
    logging.warning(f"Unknown symbol format detected: {sample_symbol}")
    return {
        'pattern': None,
        'example': sample_symbol,
        'format_type': 'unknown',
        'expiry_pattern': None,
        'description': f'Unknown format: {sample_symbol}',
        'components': None
    }


def get_adaptive_regex_pattern(format_info: Dict) -> Tuple[str, str]:
    """
    Get regex patterns for parsing symbols based on detected format.
    
    Returns:
        Tuple of (primary_pattern, fallback_pattern)
    """
    format_type = format_info.get('format_type', 'unknown')
    
    if format_type == 'full_month':
        # Primary: full month (25NOV), Fallback: abbreviated (25N25)
        return (
            r"NSE:[A-Z]+(\d{2}[A-Z]{3})(\d{5})(CE|PE)",
            r"NSE:[A-Z]+(\d{2}[A-Z]\d{2})(\d{5})(CE|PE)"
        )
    elif format_type == 'abbreviated_month':
        # Primary: abbreviated (25N25), Fallback: full month (25NOV)
        return (
            r"NSE:[A-Z]+(\d{2}[A-Z]\d{2})(\d{5})(CE|PE)",
            r"NSE:[A-Z]+(\d{2}[A-Z]{3})(\d{5})(CE|PE)"
        )
    elif format_type == 'dash_separated':
        # Dash-separated format
        return (
            r"(?i)NSE:[A-Z]+-(\d{2}-[A-Za-z]{3}-\d{4})-(\d{5})-(CE|PE)",
            None  # No fallback
        )
    elif format_type == 'dec_dcode':
        # December/weekly D-code format
        return (
            r"(?i)NSE:[A-Z]+(\d{2}[dD]\d{2})(\d{5,6})(CE|PE)",
            None
        )
    else:
        # Unknown - try both common formats
        return (
            r"(?i)NSE:[A-Z]+(\d{2}[A-Z]{3})(\d{5,6})(CE|PE)",
            r"(?i)NSE:[A-Z]+(\d{2}[A-Z]\d{2})(\d{5,6})(CE|PE)"
        )


def parse_symbol_adaptive(symbol: str, format_info: Dict) -> Optional[Dict[str, any]]:
    """
    Parse a symbol using the detected format.
    
    Returns:
        Dict with parsed components or None if parsing fails
    """
    primary_pattern, fallback_pattern = get_adaptive_regex_pattern(format_info)
    
    # Try primary pattern
    match = re.match(primary_pattern, symbol)
    if match:
        return {
            'expiry': match.group(1),
            'strike': int(match.group(2)),
            'option_type': match.group(3),
            'pattern_used': 'primary'
        }
    
    # Try fallback pattern if available
    if fallback_pattern:
        match = re.match(fallback_pattern, symbol)
        if match:
            return {
                'expiry': match.group(1),
                'strike': int(match.group(2)),
                'option_type': match.group(3),
                'pattern_used': 'fallback'
            }
    
    return None


def log_format_detection(format_info: Dict):
    """Log the detected format for monitoring"""
    logging.info("="*60)
    logging.info("FYERS SYMBOL FORMAT DETECTION")
    logging.info("="*60)
    logging.info(f"Format Type: {format_info.get('format_type', 'unknown').upper()}")
    logging.info(f"Description: {format_info.get('description', 'N/A')}")
    logging.info(f"Example Symbol: {format_info.get('example', 'N/A')}")
    logging.info(f"Expiry Pattern: {format_info.get('expiry_pattern', 'N/A')}")
    
    if format_info.get('components'):
        logging.info("Parsed Components:")
        for key, value in format_info['components'].items():
            logging.info(f"  {key}: {value}")
    
    logging.info("="*60)
