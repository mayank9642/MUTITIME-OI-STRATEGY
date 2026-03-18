import pandas as pd
import logging

def fetch_option_oi_fyers(fyers, symbol: str = "NSE:NIFTY-INDEX", strikecount: int = 20) -> pd.DataFrame:
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
        if response.get("s") != "ok" or "data" not in response or "optionsChain" not in response["data"]:
            logging.error(f"Fyers option chain fetch failed: {response}")
            return pd.DataFrame()
        records = []
        for item in response["data"]["optionsChain"]:
            if item.get("option_type") not in ("CE", "PE"):
                continue
            records.append({
                "option_type": item.get("option_type"),
                "strike": item.get("strike_price"),
                "symbol": item.get("symbol"),
                "oi": item.get("oi", 0),
                "change_oi": item.get("oich", 0),
                "ltp": item.get("ltp", 0)
            })
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch option OI data from Fyers: {e}")
        return pd.DataFrame()
