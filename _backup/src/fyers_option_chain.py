def fetch_fyers_option_chain(symbol="NIFTY"):
    """Fetch option chain as DataFrame for fallback compatibility using Fyers documented API."""
    import requests
    import yaml
    try:
        # Load credentials
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        access_token = config['fyers']['access_token']
        client_id = config['fyers']['client_id']
        # Fyers API endpoint for option chain (as per docs)
        url = "https://api.fyers.in/api/v3/option_chain"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        # Fyers uses symbols like NSE:NIFTY50-INDEX, NSE:BANKNIFTY50-INDEX
        fyers_symbol = f"NSE:{symbol}50-INDEX"
        payload = {
            "symbol": fyers_symbol,
            "expiry": "",  # Get latest expiry
            "strikeCount": 20  # Number of strikes (adjust as needed)
        }
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        logger.info(f"Fyers API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Fyers API error: HTTP {response.status_code} - {response.text}")
            return pd.DataFrame()
        try:
            data = response.json()
        except Exception as json_err:
            logger.error(f"Error decoding Fyers API JSON: {json_err}")
            logger.error(f"Raw response text: {response.text}")
            return pd.DataFrame()
        logger.info(f"Fyers API response: {data}")
        if data.get('code') != 200 or data.get('s') != 'ok':
            logger.error(f"Fyers API error: {data}")
            return pd.DataFrame()
        # Parse response to DataFrame
        options_chain = data.get('data', {}).get('optionsChain', [])
        expiry_dates = data.get('data', {}).get('expiryData', [])
        expiry_str = expiry_dates[0]['date'] if expiry_dates else ''
        rows = []
        for option in options_chain:
            strike = option.get('strike_price')
            option_type = option.get('option_type')
            ltp = option.get('ltp', 0)
            oi = option.get('oi', 0)
            oich = option.get('oich', 0)
            if option_type in ['CE', 'PE']:
                rows.append({
                    'option_type': option_type,
                    'strike': strike,
                    'symbol': f"NSE:{symbol}-{expiry_str}-{strike}-{option_type}",
                    'expiry': expiry_str,
                    'oi': oi,
                    'change_oi': oich,
                    'ltp': ltp
                })
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        logger.error(f"Error in fetch_fyers_option_chain: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()
#!/usr/bin/env python3
"""
Fyers API Option Chain Fetcher
Uses the official Fyers optionchain API to get real option chain data with OI
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
import yaml
import time

# Import Fyers API v3
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("⚠️ fyers_apiv3 not found. Install with: pip install fyers_apiv3")
    fyersModel = None

logger = logging.getLogger(__name__)

class FyersOptionChainFetcher:
    def __init__(self):
        self.fyers = None
        self.load_credentials()
    
    def load_credentials(self):
        """Load Fyers credentials and initialize API"""
        try:
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            access_token = config['fyers']['access_token']
            client_id = config['fyers']['client_id']
            
            # Check if token is still valid
            token_expiry = config['fyers'].get('token_expiry', '')
            if token_expiry:
                expiry_dt = datetime.strptime(token_expiry, '%Y-%m-%d %H:%M:%S')
                if datetime.now() > expiry_dt:
                    logger.warning("⚠️ Fyers token may be expired")
            
            # Initialize Fyers API client
            if fyersModel:
                self.fyers = fyersModel.FyersModel(
                    client_id=client_id, 
                    token=access_token, 
                    is_async=False, 
                    log_path=""
                )
                logger.info("Fyers API client initialized successfully")
            else:
                logger.error("❌ Fyers API library not available")
                raise ImportError("fyers_apiv3 library not installed")
            
        except Exception as e:
            logger.error(f"Failed to load Fyers credentials: {e}")
            raise
    
    def fetch_option_chain(self, symbol="NSE:NIFTY50-INDEX", strikecount=10):
        """
        Fetch option chain using Fyers optionchain API
        
        Args:
            symbol: Index symbol (default: NSE:NIFTY50-INDEX) 
            strikecount: Number of strikes to get (max 50)
        
        Returns:
            dict: Option chain data in standard format
        """
        try:
            logger.info(f"Fetching REAL {symbol} option chain from Fyers API...")
            
            # Prepare optionchain request
            data = {
                "symbol": symbol,
                "strikecount": strikecount,
                "timestamp": ""  # Latest data
            }
            
            logger.info(f"Requesting option chain for {symbol} with {strikecount} strikes...")
            
            # Call Fyers optionchain API
            response = self.fyers.optionchain(data=data)
            
            if response.get('code') != 200 or response.get('s') != 'ok':
                logger.error(f"Fyers API error: {response}")
                return None
            
            # Process the response to standard format
            processed_data = self.process_optionchain_response(response)
            
            if not processed_data:
                logger.error("❌ Failed to process option chain data")
                return None
                
            option_count = len(processed_data.get('records', {}).get('data', []))
            logger.info(f"Successfully fetched option chain with {option_count} options")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching Fyers option chain: {e}")
            return None
    
    def process_optionchain_response(self, response):
        """
        Process Fyers optionchain API response to standard format expected by strategy
        
        Args:
            response: Raw Fyers optionchain API response
            
        Returns:
            dict: Processed option chain in NSE format
        """
        try:
            if not response or response.get('code') != 200:
                logger.error("Invalid Fyers response")
                return None
            
            data = response.get('data', {})
            options_chain = data.get('optionsChain', [])
            
            if not options_chain:
                logger.error("No options chain data in response")
                return None
            
            # Extract underlying data (first item is usually the index)
            underlying_data = None
            option_data = []
            
            for item in options_chain:
                if item.get('option_type') == '':  # This is the underlying index
                    underlying_data = item
                else:  # This is option data
                    option_data.append(item)
            
            if not underlying_data:
                logger.error("No underlying data found")
                return None
            
            # Group options by strike price
            strikes = {}
            for option in option_data:
                strike = option.get('strike_price')
                if strike and strike > 0:  # Valid strike
                    if strike not in strikes:
                        strikes[strike] = {'CE': None, 'PE': None}
                    
                    option_type = option.get('option_type')
                    if option_type in ['CE', 'PE']:
                        strikes[strike][option_type] = {
                            'strikePrice': strike,
                            'openInterest': option.get('oi', 0),
                            'changeinOpenInterest': option.get('oich', 0),
                            'pchangeinOpenInterest': option.get('oichp', 0.0),
                            'totalTradedVolume': option.get('volume', 0),
                            'impliedVolatility': 0.0,  # Not provided by Fyers
                            'lastPrice': option.get('ltp', 0.0),
                            'change': option.get('ltpch', 0.0),
                            'pChange': option.get('ltpchp', 0.0),
                            'totalBuyQuantity': 0,  # Not provided
                            'totalSellQuantity': 0,  # Not provided
                            'bidQty': 0,  # Not provided
                            'bidprice': option.get('bid', 0.0),
                            'askQty': 0,  # Not provided
                            'askPrice': option.get('ask', 0.0),
                            'underlying': underlying_data.get('ltp', 0.0)
                        }
            
            # Convert to list format expected by strategy
            processed_options = []
            for strike in sorted(strikes.keys()):
                strike_data = strikes[strike]
                
                option_entry = {
                    'strikePrice': strike,
                    'expiryDate': data.get('expiryData', [{}])[0].get('date', ''),
                }
                
                # Add CE data
                if strike_data['CE']:
                    option_entry['CE'] = strike_data['CE']
                
                # Add PE data  
                if strike_data['PE']:
                    option_entry['PE'] = strike_data['PE']
                
                processed_options.append(option_entry)
            
            # Build final response structure
            result = {
                'records': {
                    'data': processed_options,
                    'timestamp': datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                    'underlyingValue': underlying_data.get('ltp', 0.0),
                    'strikePrices': list(strikes.keys())
                },
                'filtered': {
                    'data': processed_options[:20]  # Limit for performance
                },
                'metadata': {
                    'callOI': data.get('callOi', 0),
                    'putOI': data.get('putOi', 0),
                    'expiryDates': [exp.get('date', '') for exp in data.get('expiryData', [])],
                    'indiaVIX': data.get('indiavixData', {}).get('ltp', 0.0)
                }
            }
            
            logger.info(f"Processed {len(processed_options)} options from Fyers data")
            return result
            
        except Exception as e:
            logger.error(f"Error processing Fyers optionchain response: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_option_chain_dataframe(self, symbol="NIFTY"):
        """Get option chain as pandas DataFrame with required columns for OI analysis, using Fyers-compliant symbol formatting."""
        from src.symbol_utils import convert_option_symbol_format
        chain_data = self.fetch_option_chain(f"NSE:{symbol}50-INDEX")
        if not chain_data:
            return pd.DataFrame()
        try:
            options = chain_data.get('records', {}).get('data', [])
            expiry_from_meta = ''
            expiry_dates = chain_data.get('metadata', {}).get('expiryDates', [])
            if expiry_dates and isinstance(expiry_dates, list) and len(expiry_dates) > 0:
                expiry_from_meta = expiry_dates[0]
            rows = []
            for option in options:
                strike = option.get('strikePrice')
                expiry = option.get('expiry', '') or expiry_from_meta
                # CE data
                if 'CE' in option:
                    ce = option['CE']
                    raw_symbol = f"NSE:{symbol}-{expiry}-{strike}-CE"
                    rows.append({
                        'symbol': convert_option_symbol_format(raw_symbol),
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': 'CE',
                        'ltp': ce.get('lastPrice', 0),
                        'oi': ce.get('openInterest', 0),
                        'change_oi': ce.get('changeinOpenInterest', 0),
                        'volume': ce.get('totalTradedVolume', 0)
                    })
                # PE data
                if 'PE' in option:
                    pe = option['PE']
                    raw_symbol = f"NSE:{symbol}-{expiry}-{strike}-PE"
                    rows.append({
                        'symbol': convert_option_symbol_format(raw_symbol),
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': 'PE',
                        'ltp': pe.get('lastPrice', 0),
                        'oi': pe.get('openInterest', 0),
                        'change_oi': pe.get('changeinOpenInterest', 0),
                        'volume': pe.get('totalTradedVolume', 0)
                    })
            df = pd.DataFrame(rows)
            return df
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def get_option_chain_data(self, symbols, max_symbols=50):
        """Get quotes for multiple option symbols using individual requests"""
        try:
            logger.info(f"📊 Fetching option chain data for {len(symbols)} symbols (individual requests)...")
            
            all_quotes = []
            valid_symbols = []
            
            # Limit to reasonable number to avoid too many API calls
            symbols_to_test = symbols[:max_symbols]
            logger.info(f"Testing first {len(symbols_to_test)} symbols...")
            
            for i, symbol in enumerate(symbols_to_test):
                try:
                    if i % 10 == 0:
                        logger.info(f"Progress: {i+1}/{len(symbols_to_test)}")
                    
                    url = f"{self.base_url}/quotes"
                    params = {'symbols': symbol}
                    headers = self.get_headers()
                    
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('s') == 'ok' and data.get('d'):
                            quote = data['d'][0]
                            
                            if quote.get('s') == 'ok':
                                # Check if quote has valid data
                                price_data = quote.get('v', {})
                                ltp = price_data.get('lp', 0)
                                volume = price_data.get('volume', 0)
                                
                                if ltp > 0:  # Valid price
                                    all_quotes.append(quote)
                                    valid_symbols.append(symbol)
                                    
                                    if len(valid_symbols) % 5 == 0:
                                        logger.debug(f"Found {len(valid_symbols)} valid options so far...")
                                
                    elif response.status_code == 500:
                        # Symbol doesn't exist, skip
                        continue
                    else:
                        logger.debug(f"Symbol {symbol}: HTTP {response.status_code}")
                    
                    # Small delay between requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    continue
            
            logger.info(f"✅ Retrieved {len(all_quotes)} valid option quotes from {len(valid_symbols)} symbols")
            
            if len(all_quotes) > 0:
                logger.info("Sample valid symbols:")
                for i, sym in enumerate(valid_symbols[:5]):
                    logger.info(f"  {sym}")
            
            return all_quotes
            
        except Exception as e:
            logger.error(f"Error getting option chain data: {e}")
            return []
    
    def process_to_option_chain(self, quotes):
        """Convert Fyers quotes to option chain DataFrame"""
        try:
            logger.info("🔄 Processing Fyers data to option chain format...")
            
            records = []
            
            for quote in quotes:
                symbol = quote.get('n', '')
                
                if 'NIFTY' not in symbol:
                    continue
                
                try:
                    # Parse symbol: NSE:NIFTY25L1226000CE
                    if symbol.endswith('CE'):
                        option_type = 'CE'
                        # Remove NSE:NIFTY and CE to get date+strike: 25L1226000
                        middle_part = symbol.replace('NSE:NIFTY', '').replace('CE', '')
                        
                    elif symbol.endswith('PE'):
                        option_type = 'PE'
                        # Remove NSE:NIFTY and PE to get date+strike: 25L1226000
                        middle_part = symbol.replace('NSE:NIFTY', '').replace('PE', '')
                    else:
                        continue
                    
                    # Parse middle_part: 25D0926050 (year + month_letter + day + strike)
                    if len(middle_part) >= 8:  # At least 2+1+2+5 = 10 chars expected
                        year = 2000 + int(middle_part[:2])  # 25 -> 2025
                        month_letter = middle_part[2]       # D
                        
                        # Map month letter back to number
                        month_map = {
                            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
                            'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12
                        }
                        month = month_map.get(month_letter, 12)
                        
                        # Rest is day + strike: 0926050 -> day=09, strike=26050
                        remaining = middle_part[3:]
                        if len(remaining) >= 7:  # 2 digits day + 5 digits strike minimum
                            day = int(remaining[:2])      # 09
                            strike = int(remaining[2:])   # 26050 (5 digits)
                        else:
                            continue
                        
                        expiry_date = datetime(year, month, day)
                        expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
                    else:
                        continue
                    
                    # Get quote data
                    ltp = quote.get('lp', 0)
                    volume = quote.get('v', 0)
                    
                    # Use volume as proxy for OI (Fyers doesn't provide OI directly in quotes)
                    # We can also use other metrics like total traded value
                    oi_proxy = volume * ltp  # Volume * Price as liquidity indicator
                    
                    records.append({
                        'option_type': option_type,
                        'strike': strike,
                        'symbol': symbol,
                        'expiry': expiry_str,
                        'oi': int(oi_proxy),  # Using volume*price as OI proxy
                        'change_oi': 0,  # Not available in quotes
                        'ltp': ltp,
                        'volume': volume  # Keep original volume for reference
                    })
                    
                except Exception as e:
                    logger.debug(f"Error parsing symbol {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(records)
            
            if len(df) > 0:
                # Sort by OI proxy (volume * price) descending
                df = df.sort_values('oi', ascending=False).reset_index(drop=True)
                
                # Filter for reasonable strikes (remove very low volume options)
                min_oi = df['oi'].quantile(0.1)  # Keep top 90% by liquidity
                df = df[df['oi'] >= min_oi].reset_index(drop=True)
                
                logger.info(f"✅ Processed {len(df)} liquid option contracts")
                
                # Show summary
                ce_count = len(df[df['option_type'] == 'CE'])
                pe_count = len(df[df['option_type'] == 'PE'])
                logger.info(f"CE options: {ce_count}, PE options: {pe_count}")
                
                if len(df) > 0:
                    logger.info(f"Strike range: {df['strike'].min()} - {df['strike'].max()}")
                    logger.info(f"Expiries: {df['expiry'].unique()}")
                    
                    # Show top liquid options
                    logger.info("📊 Most liquid options:")
                    top_5 = df.head(5)
                    for _, row in top_5.iterrows():
                        logger.info(f"  {row['strike']} {row['option_type']}: LTP={row['ltp']}, Vol={row['volume']}")
            
            else:
                logger.warning("⚠️ No valid options processed from Fyers data")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing Fyers quotes: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
# Global fetcher instance
_fyers_fetcher = None

def fetch_option_oi(symbol="NIFTY"):
    """Fetch real option chain data using Fyers API"""
    global _fyers_fetcher
    
    try:
        if _fyers_fetcher is None:
            _fyers_fetcher = FyersOptionChainFetcher()
        
        df = _fyers_fetcher.fetch_option_chain(symbol)
        
        if len(df) > 0:
            logger.info("✅ SUCCESS: Using REAL Fyers option chain data!")
            return df
        else:
            logger.error("❌ Failed to get Fyers option chain data")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in Fyers fetch_option_oi: {e}")
        return pd.DataFrame()

def get_option_chain_data(symbol="NIFTY"):
    """Get option chain data in dict format (global function for strategy)"""
    global _fyers_fetcher
    
    try:
        if _fyers_fetcher is None:
            _fyers_fetcher = FyersOptionChainFetcher()
        
        return _fyers_fetcher.fetch_option_chain(f"NSE:{symbol}50-INDEX")
        
    except Exception as e:
        logger.error(f"Error in get_option_chain_data: {e}")
        return None

def test_fyers_optionchain():
    """Test Fyers optionchain API"""
    print("🧪 Testing Fyers Option Chain Fetcher...")
    
    try:
        fetcher = FyersOptionChainFetcher()
        result = fetcher.fetch_option_chain("NSE:NIFTY50-INDEX", strikecount=5)
        
        if result:
            print("✅ Successfully fetched option chain!")
            print(f"📊 Options count: {len(result.get('records', {}).get('data', []))}")
            print(f"🎯 Underlying: {result.get('records', {}).get('underlyingValue', 0)}")
            print(f"📈 Call OI: {result.get('metadata', {}).get('callOI', 0)}")
            print(f"📉 Put OI: {result.get('metadata', {}).get('putOI', 0)}")
        else:
            print("❌ Failed to fetch option chain")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_fyers_optionchain()