"""
Utility functions to convert option symbols to the format required by Fyers API
"""
import logging
import re
import datetime

def convert_option_symbol_format(symbol):
    """
    Convert option symbols to the format required by Fyers API
    Example: NSE:NIFTY-04-Nov-2025-27450-CE → NSE:NIFTY04NOV2527450CE
    """
    if not symbol:
        return symbol
    # If not an option symbol, return as is
    if "CE" not in symbol and "PE" not in symbol:
        return symbol
    # If already canonical (no hyphens, matches Fyers format), return as is
    if re.match(r"^(NSE:)?(NIFTY|BANKNIFTY)\d{2}[A-Z]{3}\d{2,4}\d{5}(CE|PE)$", symbol):
        # Already in correct format
        if not symbol.startswith("NSE:"):
            return "NSE:" + symbol
        return symbol
    try:
        # Accepts formats like: NIFTY-27-FEB-2026-26500-CE, NIFTY_27_FEB_2026_26500_CE, etc.
        # Normalize separators
        s = symbol.replace("_", "-").replace(" ", "-")
        prefix = "NSE:"
        rest = s
        if ":" in s:
            parts = s.split(":")
            prefix = parts[0] + ":"
            rest = parts[1]
        components = rest.split("-")
        # Try to extract underlying, day, month, year, strike, option_type
        if len(components) >= 6:
            underlying = components[0].upper()
            day = components[1].zfill(2)
            month = components[2].upper()[:3]
            year = components[3]
            strike = components[4].zfill(5)
            option_type = components[5].upper()
            if len(year) == 4:
                year = year[2:]
            expiry = f"{day}{month}{year}"
            return f"{prefix}{underlying}{expiry}{strike}{option_type}"
        # Fallback: try regex extraction
        m = re.match(r"(NIFTY|BANKNIFTY)[-_]?(\d{1,2})[-_]?([A-Z]{3})[-_]?((?:\d{2})|(?:\d{4}))[-_]?(\d{4,5})[-_]?([CP]E)", rest, re.IGNORECASE)
        if m:
            underlying, day, month, year, strike, option_type = m.groups()
            day = day.zfill(2)
            month = month.upper()[:3]
            if len(year) == 4:
                year = year[2:]
            strike = strike.zfill(5)
            option_type = option_type.upper()
            expiry = f"{day}{month}{year}"
            return f"{prefix}{underlying.upper()}{expiry}{strike}{option_type}"
        # If all else fails, return original
        return symbol
    except Exception as e:
        print(f"Error converting option symbol {symbol}: {e}")
        return symbol
