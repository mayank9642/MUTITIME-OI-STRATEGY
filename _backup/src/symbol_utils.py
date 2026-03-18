"""
Combined symbol generation and matching utilities.

Paste this single file into your VS Code workspace (e.g., src/symbol_utils_for_vscode.py)
and use it with your project. It contains:
- convert_option_symbol_format(symbol): converts various option symbol formats into Fyers API format
- get_canonical_symbol(symbol): canonicalizer extracted from strategy.py (standalone version)
- patched_get_option_chain(original_get_option_chain): helper wrapper to convert DataFrame 'symbol' column
"""

import re
import logging
import datetime
from typing import Callable

# -----------------------------------------------------------
# convert_option_symbol_format
# -----------------------------------------------------------
def convert_option_symbol_format(symbol: str) -> str:
    """
    Convert option symbols to the format required by Fyers API.

    Expected Fyers format (example):
      NSE:NIFTY25JUL24700PE

    This function tries to handle:
    - Hyphenated exchange-style symbols like "NIFTY-28-JUL-25-24700-CE" or "NIFTY-28JUL25-24700-CE"
    - Colon prefixed ones like "NSE:NIFTY-28JUL25-24700-CE"
    - Already-correct symbols are returned unchanged.

    Returns the converted symbol or the original symbol if conversion fails.
    """
    if not symbol:
        return symbol

    # If it's not an option symbol (no CE/PE), return as is
    if "CE" not in symbol and "PE" not in symbol and not re.search(r"[CP]E$|[CP]$", symbol):
        return symbol

    # If already looks like Fyers format (has colon and no hyphen), assume OK
    if ":" in symbol and "-" not in symbol:
        return symbol

    logging.debug(f"Converting symbol: {symbol}")

    try:
        prefix = ""
        rest = symbol

        # Handle optional exchange prefix like "NSE:"
        if ":" in symbol:
            parts = symbol.split(":")
            prefix = parts[0].upper() + ":"
            rest = parts[1]

        # Normalize separators to hyphen for easier splitting
        normalized = re.sub(r'[_\s]+', '-', rest)
        # If there are embedded parentheses or other noise, strip them
        normalized = re.sub(r'[()\[\]]', '', normalized)

        components = normalized.split("-")

        # Some symbols might come without hyphens, e.g., NIFTY07AUG25C24550
        # In that case, try to detect with regex
        if len(components) == 1:
            # Try NIFTY07AUG25C24550 or NIFTY07AUG25P24550
            m = re.match(r'([A-Z]+)(\d{2})([A-Z]{3})(\d{2})([CP])(\d+)', components[0], re.IGNORECASE)
            if m:
                underlying = m.group(1).upper()
                day = m.group(2)
                month = m.group(3).upper()
                year = m.group(4)
                opt_type_letter = m.group(5).upper()
                strike_price = m.group(6)
                option_type = "CE" if opt_type_letter == "C" else "PE"
                fyers = f"{prefix}{underlying}{year}{month}{day}{strike_price}{option_type}"
                logging.debug(f"Converted compact form: {components[0]} -> {fyers}")
                return fyers

        # Otherwise, parse components
        underlying = components[0].upper() if components else rest.upper()

        # Find option type CE/PE in components (case-insensitive)
        option_type = None
        for part in components[::-1]:
            pl = part.upper()
            if pl in ("CE", "PE"):
                option_type = pl
                break
            # some inputs might have C or P
            if pl in ("C", "P"):
                option_type = "CE" if pl == "C" else "PE"
                break

        if not option_type:
            logging.warning(f"Could not find option type (CE/PE) in symbol: {symbol}")
            # give it a chance: if symbol ends with CE/PE without separator
            m_end = re.search(r'(CE|PE)$', rest, re.IGNORECASE)
            if m_end:
                option_type = m_end.group(1).upper()
            else:
                return symbol

        # Strike price: look for a numeric part (usually 4-6 digits)
        strike_price = None
        for part in components[::-1]:
            p = re.sub(r'\D', '', part)  # digits only
            if p and len(p) >= 3:  # allow 3+ digits (e.g., 700)
                strike_price = p
                break

        if not strike_price:
            # try extracting digits right before CE/PE
            m_strike = re.search(r'(\d{3,6})(?:CE|PE|C|P)?$', rest, re.IGNORECASE)
            if m_strike:
                strike_price = m_strike.group(1)

        if not strike_price:
            logging.warning(f"Could not find strike price in symbol: {symbol}")
            return symbol

        # Date pieces: day (1-31), month (JAN..DEC), year (2 or 4 digits)
        day = None
        month = None
        year = None

        months_list = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

        # Look for day
        for part in components:
            if part.isdigit() and len(part) == 2 and 1 <= int(part) <= 31:
                day = part
                break

        # Look for month abbreviation
        for part in components:
            pu = part.upper()
            if pu in months_list:
                month = pu
                break
            # Sometimes month and day are combined like 28JUL or JUL28
            m_combo = re.match(r'(\d{1,2})([A-Z]{3})', part, re.IGNORECASE)
            if m_combo and m_combo.group(2).upper() in months_list:
                day = m_combo.group(1).zfill(2)
                month = m_combo.group(2).upper()
                break
            m_combo2 = re.match(r'([A-Z]{3})(\d{1,2})', part, re.IGNORECASE)
            if m_combo2 and m_combo2.group(1).upper() in months_list:
                month = m_combo2.group(1).upper()
                day = m_combo2.group(2).zfill(2)
                break

        # Look for year (2-digit or 4-digit)
        for part in components[::-1]:
            if part.isdigit() and (len(part) == 2 or len(part) == 4):
                if len(part) == 4:
                    year = part[2:]
                else:
                    year = part
                # avoid treating day as year
                if part != day:
                    break

        # If missing some components, fall back to defaults (today)
        if not day or not month or not year:
            logging.info(f"Missing date component(s) for {symbol}. Using today's date for missing parts.")
            today = datetime.datetime.now()
            day = day or today.strftime('%d')
            month = month or today.strftime('%b').upper()
            year = year or today.strftime('%y')

        # Fyers expects expiry as YYMDD (year, single-letter month, day)
        # Month mapping for Fyers single-letter codes
        month_map = {
            "JAN": "A", "FEB": "B", "MAR": "C", "APR": "D", "MAY": "E", "JUN": "F",
            "JUL": "G", "AUG": "H", "SEP": "I", "OCT": "J", "NOV": "N", "DEC": "D"
        }
        fyers_month = month_map.get(month, month[0]) if month else ""
        new_symbol = f"{prefix}{underlying}{year}{fyers_month}{day}{strike_price}{option_type}"
        logging.debug(f"Converted: {symbol} -> {new_symbol}")
        return new_symbol

    except Exception as e:
        logging.error(f"Error converting option symbol {symbol}: {e}")
        return symbol

# -----------------------------------------------------------
# get_canonical_symbol
# -----------------------------------------------------------
def get_canonical_symbol(symbol: str) -> str:
    """
    Convert any incoming symbol to a canonical Fyers-like format.

    This is a standalone adaptation of the logic in src/strategy.py.
    It tries several patterns:
    - If symbol already starts with 'NSE:' and ends with CE/PE -> return as is.
    - Try compact NIFTY pattern: NIFTY07AUG25C24550 or NIFTY07AUG25P24550
    - Fallback to convert_option_symbol_format if pattern doesn't match.
    """
    orig_symbol = symbol
    try:
        # If already in Fyers format, return as is
        if isinstance(symbol, str) and symbol.startswith('NSE:') and (symbol.endswith('CE') or symbol.endswith('PE')):
            logging.info(f"[SYMBOL MAP] Already canonical: {symbol}")
            return symbol

        # Try to match NIFTY compact options: NIFTY07AUG25C24550 (C/P single letter)
        m = re.match(r'([A-Z]+)(\d{2})([A-Z]{3})(\d{2})([CP])(\d+)$', symbol, re.IGNORECASE)
        if m:
            underlying = m.group(1).upper()
            day = m.group(2)
            month = m.group(3).upper()
            year = m.group(4)
            opt_type_letter = m.group(5).upper()
            strike = m.group(6)
            opt_type = 'CE' if opt_type_letter == 'C' else 'PE'
            fyers_symbol = f"NSE:{underlying}{year}{month}{day}{strike}{opt_type}"
            logging.info(f"[SYMBOL MAP] {orig_symbol} → {fyers_symbol}")
            return fyers_symbol

        # Try alternate pattern with CE/PE suffix and strike: e.g., NIFTY07AUG25C24550 or NIFTY07AUG25P24550
        m2 = re.match(r'([A-Z]+)(\d{2})([A-Z]{3})(\d{2})(CE|PE)(\d+)$', symbol, re.IGNORECASE)
        if m2:
            underlying = m2.group(1).upper()
            day = m2.group(2)
            month = m2.group(3).upper()
            year = m2.group(4)
            opt_type = m2.group(5).upper()
            strike = m2.group(6)
            fyers_symbol = f"NSE:{underlying}{year}{month}{day}{strike}{opt_type}"
            logging.info(f"[SYMBOL MAP] {orig_symbol} → {fyers_symbol}")
            return fyers_symbol

        # Fallback: try convert_option_symbol_format
        converted = convert_option_symbol_format(symbol)
        logging.info(f"[SYMBOL MAP] {orig_symbol} → {converted}")
        return converted

    except Exception as e:
        logging.error(f"[SYMBOL MAP] Error converting {orig_symbol}: {e}")
        return symbol

# -----------------------------------------------------------
# patched_get_option_chain helper
# -----------------------------------------------------------
def patched_get_option_chain(original_get_option_chain: Callable):
    """
    Return a wrapper function that calls original_get_option_chain and
    converts the 'symbol' column in any returned pandas DataFrame using
    convert_option_symbol_format.

    Usage:
      # in your run script:
      import src.nse_data_new as nse
      nse.get_nifty_option_chain = patched_get_option_chain(nse.get_nifty_option_chain)

    Note: this function requires pandas at runtime (only imported inside wrapper).
    """
    def wrapper(*args, **kwargs):
        result = original_get_option_chain(*args, **kwargs)
        try:
            import pandas as pd
        except Exception:
            logging.warning("pandas not available; returning original result")
            return result

        # If we got a DataFrame with a 'symbol' column, transform it
        if isinstance(result, pd.DataFrame) and 'symbol' in result.columns:
            if not result.empty:
                logging.info("Original option symbols (sample):")
                for i, s in enumerate(result['symbol'].iloc[:5]):
                    logging.info(f"  {i+1}. {s}")

            logging.info("Converting option symbols to Fyers API format")
            result['symbol'] = result['symbol'].apply(convert_option_symbol_format)

            if not result.empty:
                logging.info("Converted option symbols (sample):")
                for i, s in enumerate(result['symbol'].iloc[:5]):
                    logging.info(f"  {i+1}. {s}")

        return result

    return wrapper

# -----------------------------------------------------------
# Example quick script usage when pasted into project:
# -----------------------------------------------------------
if __name__ == "__main__":
    # small self-test examples
    examples = [
        "NIFTY-28-JUL-25-24700-CE",
        "NSE:NIFTY-28JUL25-24700-CE",
        "NIFTY07AUG25C24550",
        "NIFTY07AUG25P24550",
        "NIFTY-07-AUG-25-24550-PE",
        "BANKNIFTY-10SEP25-77000-PE",
    ]
    for s in examples:
        try:
            print(f"{s}  ->  {convert_option_symbol_format(s)}")
        except Exception as e:
            print(f"Error converting {s}: {e}")

    # canonical examples
    canon_examples = [
        "NIFTY07AUG25C24550",
        "NIFTY07AUG25P24550",
        "NSE:NIFTY28JUL2524700PE",
    ]
    for s in canon_examples:
        print(f"canonical({s}) -> {get_canonical_symbol(s)}")
