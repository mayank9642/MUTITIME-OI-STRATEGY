import pandas as pd
import logging

def fetch_option_oi_fyers(fyers, symbol: str = "NSE:NIFTY50-INDEX", strikecount: int = 20) -> pd.DataFrame:
    """
    Fetch option chain data from Fyers and return a DataFrame with OI details.
    Args:
        fyers: Authenticated Fyers client
        symbol (str): Fyers symbol (e.g., 'NSE:NIFTY-INDEX')
        strikecount (int): Number of strikes to fetch (ATM, ITM, OTM)
    Returns:
        pd.DataFrame: DataFrame containing option_type, strike, symbol, oi, change_oi, ltp for calls and puts.
    """
    try:
        data = {
            "symbol": symbol,
            "strikecount": strikecount,
            "timestamp": ""
        }
        response = fyers.optionchain(data=data)
        # Log full response at DEBUG for troubleshooting if something's unexpected
        try:
            logging.debug(f"Raw optionchain response for {symbol}: {response}")
        except Exception:
            pass

        if not isinstance(response, dict) or response.get("s") != "ok":
            logging.error(f"Fyers option chain fetch failed or returned non-ok status: {response}")
            return pd.DataFrame()

        # Ensure optionsChain exists in data; if not, log details for diagnostics
        data_block = response.get('data') or {}
        if 'optionsChain' not in data_block or not data_block.get('optionsChain'):
            logging.debug(f"Optionchain API returned no 'optionsChain' key or empty list. expiryData: {data_block.get('expiryData')}, indiavixData: {data_block.get('indiavixData')}")
            return pd.DataFrame()
        records = []
        for item in data_block.get("optionsChain", []):
            if item.get("option_type") not in ("CE", "PE"):
                continue
            # Try to obtain expiry explicitly if provided by the API; otherwise parse from symbol
            expiry = item.get('expiry') or item.get('expiry_date') or item.get('expiryDate')
            if not expiry:
                # Fallback: parse common patterns within the symbol, e.g., 02DEC25 or 02DEC2025 or yymmdd/yyyymmdd
                sym = item.get('symbol') or ''
                expiry = None
                try:
                    import re
                    m = re.search(r"(\d{1,2}[A-Z]{3}\d{2,4})", sym)
                    if m:
                        s = m.group(1)
                        day = s[:2]
                        mon = s[2:5]
                        year = s[5:]
                        if len(year) == 2:
                            year = '20' + year
                        try:
                            from datetime import datetime
                            dt = datetime.strptime(day + mon + year, "%d%b%Y")
                            expiry = dt.date().isoformat()
                        except Exception:
                            try:
                                dt = datetime.strptime(s, "%d%b%y")
                                expiry = dt.date().isoformat()
                            except Exception:
                                expiry = None
                    else:
                        # try numeric yyyymmdd or yymmdd groups
                        m2 = re.search(r"(\d{6,8})", sym)
                        if m2:
                            s2 = m2.group(1)
                            try:
                                from datetime import datetime
                                if len(s2) == 6:
                                    dt = datetime.strptime(s2, "%y%m%d")
                                else:
                                    dt = datetime.strptime(s2, "%Y%m%d")
                                expiry = dt.date().isoformat()
                            except Exception:
                                expiry = None
                except Exception:
                    expiry = None

            records.append({
                "option_type": item.get("option_type"),
                "strike": item.get("strike_price"),
                "symbol": item.get("symbol"),
                "oi": item.get("oi", 0),
                "change_oi": item.get("oich", 0),
                "ltp": item.get("ltp", 0),
                "expiry": expiry
            })

        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch option OI data from Fyers: {e}")
        return pd.DataFrame()
