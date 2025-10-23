"""
Utility functions to convert option symbols to the format required by Fyers API
"""
import logging
import re
import datetime

def convert_option_symbol_format(symbol):
    """
    Convert option symbols to the format required by Fyers API
    
    Based on testing, Fyers API expects option symbols in format:
    NSE:NIFTY25JUL24700PE (no hyphens)
    
    Args:
        symbol (str): Option symbol to convert
        
    Returns:
        str: Symbol in Fyers API compatible format
    """
    if not symbol:
        return symbol
        
    # If it's not an option symbol (no CE/PE), return as is
    if "CE" not in symbol and "PE" not in symbol:
        return symbol
        
    # Check if already in correct format (no hyphens)
    if "-" not in symbol and ":" in symbol:
        return symbol
    
    print(f"Converting symbol: {symbol}")
        
    try:
        # Extract exchange prefix (e.g., "NSE:")
        prefix = ""
        rest = symbol
        if ":" in symbol:
            parts = symbol.split(":")
            prefix = parts[0] + ":"
            rest = parts[1]
        
        # Extract the underlying symbol (e.g., "NIFTY")
        components = rest.split("-")
        underlying = components[0]
        
        # Find option type (CE/PE)
        option_type = None
        for part in components:
            if part == "CE" or part == "PE":
                option_type = part
                break
                
        if not option_type:
            print(f"Could not find option type (CE/PE) in symbol: {symbol}")
            return symbol
        
        # Find the strike price (usually 5 digits for NIFTY)
        strike_price = None
        for part in components:
            if part.isdigit() and len(part) >= 4:
                strike_price = part
                break
                
        if not strike_price:
            print(f"Could not find strike price in symbol: {symbol}")
            return symbol
            
        # Find date components
        day = None
        month = None
        year = None
        
        # Look for day (2-digit number between 1-31)
        for part in components:
            if part.isdigit() and len(part) == 2 and 1 <= int(part) <= 31:
                day = part
                break
                
        # Look for month abbreviation (JAN, FEB, etc.)
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for part in components:
            part_upper = part.upper()
            if part_upper in months:
                month = part_upper
                break
                
        # Look for year (2-digit or 4-digit)
        for part in components:
            if part.isdigit() and (len(part) == 2 or len(part) == 4):
                if len(part) == 4:
                    # Convert 4-digit year to 2-digit
                    year = part[2:]
                else:
                    year = part
                # Only consider as year if it's not already identified as the day
                if part != day:
                    break
        
        # Ensure we have all required components
        if not day or not month or not year:
            print(f"Missing date component in {symbol}. Using defaults.")
            today = datetime.datetime.now()
            day = day or today.strftime('%d')
            month = month or today.strftime('%b').upper()
            year = year or today.strftime('%y')
        
        # Build the final symbol in format: NSE:NIFTY28JUL2524700PE
        new_symbol = f"{prefix}{underlying}{day}{month}{year}{strike_price}{option_type}"
        
        print(f"Converted: {symbol} → {new_symbol}")
        return new_symbol
    
    except Exception as e:
        print(f"Error converting option symbol {symbol}: {e}")
        return symbol  # Return original symbol if conversion fails
