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
    # If already canonical (no hyphens), return as is
    if "-" not in symbol and ":" in symbol:
        return symbol
    try:
        # Extract exchange prefix
        prefix = ""
        rest = symbol
        if ":" in symbol:
            parts = symbol.split(":")
            prefix = parts[0] + ":"
            rest = parts[1]
        components = rest.split("-")
        # Expect format: underlying-day-month-year-strike-CE/PE
        if len(components) < 6:
            # Fallback: return original symbol
            print(f"Symbol format not recognized: {symbol}")
            return symbol
        underlying = components[0]
        day = components[1]
        month = components[2].upper()
        year = components[3]
        strike_price = components[4]
        option_type = components[5]
        # Convert 4-digit year to 2-digit
        if len(year) == 4:
            year = year[2:]
        # Build canonical symbol
        new_symbol = f"{prefix}{underlying}{day}{month}{year}{strike_price}{option_type}"
        print(f"Converted: {symbol} → {new_symbol}")
        return new_symbol
    except Exception as e:
        print(f"Error converting option symbol {symbol}: {e}")
        return symbol
