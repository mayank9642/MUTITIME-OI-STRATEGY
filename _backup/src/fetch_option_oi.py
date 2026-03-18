import requests
import pandas as pd
import logging
from datetime import datetime
from src.symbol_utils import convert_option_symbol_format

# Configure logging
logger = logging.getLogger(__name__)

# Example NSE Option Chain URL (for NIFTY, can be parameterized)
NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain"
}


def fetch_option_oi(symbol: str = "NIFTY") -> pd.DataFrame:
    """
    Fetches option chain data from NSE and returns a DataFrame with OI details.
    Args:
        symbol (str): Index symbol (e.g., 'NIFTY', 'BANKNIFTY').
    Returns:
        pd.DataFrame: DataFrame containing option_type, strike, symbol, oi, change_oi, ltp for calls and puts.
    """
    try:
        session = requests.Session()
        # Initial request to get cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        url = NSE_OPTION_CHAIN_URL.format(symbol=symbol)
        response = session.get(url, headers=NSE_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        today = pd.Timestamp(datetime.now().date())
        expiry_dates = [item.get("expiryDate", "") for item in data["records"]["data"] if item.get("expiryDate", "")]
        expiry_dates_dt = sorted(set(pd.to_datetime(d) for d in expiry_dates if d))
        # Only consider future expiry dates
        valid_expiries_dt = [d for d in expiry_dates_dt if d >= today]
        # Find next Thursday after today (weekly expiry)
        next_thursday = None
        for d in valid_expiries_dt:
            if d.weekday() == 3 and d > today:
                next_thursday = d
                break
        # Find last Thursday of current month (monthly expiry)
        month = today.month
        year = today.year
        thursdays_in_month = [d for d in valid_expiries_dt if d.month == month and d.year == year and d.weekday() == 3]
        last_thursday = max(thursdays_in_month) if thursdays_in_month else None
        # Prepare expiry strings
        expiry_strs = set()
        if next_thursday:
            expiry_strs.add(next_thursday.strftime("%d-%b-%Y"))
        if last_thursday:
            expiry_strs.add(last_thursday.strftime("%d-%b-%Y"))
        records = []
        for item in data["records"]["data"]:
            strike = item.get("strikePrice")
            expiry = item.get("expiryDate", "")
            if expiry_strs and expiry not in expiry_strs:
                continue  # Skip non-tradable expiry
            ce = item.get("CE", {})
            pe = item.get("PE", {})
            # Add CE row
            if ce:
                raw_contract_symbol = f"NSE:{symbol}-{ce.get('expiryDate','')}-{strike}-CE"
                contract_symbol = convert_option_symbol_format(raw_contract_symbol)
                records.append({
                    "option_type": "CE",
                    "strike": strike,
                    "symbol": contract_symbol,
                    "oi": ce.get("openInterest", 0),
                    "change_oi": ce.get("changeinOpenInterest", 0),
                    "ltp": ce.get("lastPrice", 0)
                })
            # Add PE row
            if pe:
                raw_contract_symbol = f"NSE:{symbol}-{pe.get('expiryDate','')}-{strike}-PE"
                contract_symbol = convert_option_symbol_format(raw_contract_symbol)
                records.append({
                    "option_type": "PE",
                    "strike": strike,
                    "symbol": contract_symbol,
                    "oi": pe.get("openInterest", 0),
                    "change_oi": pe.get("changeinOpenInterest", 0),
                    "ltp": pe.get("lastPrice", 0)
                })
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch option OI data: {e}")
        return pd.DataFrame()
