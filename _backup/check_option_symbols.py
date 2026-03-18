#!/usr/bin/env python3
"""
Check current NIFTY price and test option symbol formats
"""
import requests
import yaml

def check_nifty_and_options():
    """Check NIFTY price and test option symbols"""
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    access_token = config['fyers']['access_token']
    client_id = config['fyers']['client_id']
    
    headers = {
        'Authorization': f'{client_id}:{access_token}',
    }
    
    base_url = "https://api.fyers.in/api/v2"
    
    # Get NIFTY price
    print("🔍 Getting current NIFTY price...")
    try:
        url = f"{base_url}/quotes"
        params = {'symbols': 'NSE:NIFTY50-INDEX'}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('s') == 'ok':
                quote = data['d'][0]['v']
                nifty_price = quote.get('lp', 0)
                print(f"✅ NIFTY Current Price: {nifty_price}")
                
                # Calculate ATM strike
                atm_strike = round(nifty_price / 50) * 50
                print(f"ATM Strike: {atm_strike}")
                
                # Test common option symbol formats around ATM
                test_strikes = [atm_strike - 100, atm_strike, atm_strike + 100]
                
                # Common option symbol formats to try
                option_formats = [
                    "NSE:NIFTY{expiry}C{strike}",      # NSE:NIFTY25D12C24500
                    "NSE:NIFTY{expiry}{strike}CE",     # NSE:NIFTY25D1224500CE  
                    "NSE:NIFTY25{month}{day}C{strike}", # NSE:NIFTY25D12C24500
                    "NSE:NIFTY{year}{month}{day}{strike}CE",  # NSE:NIFTY25121224500CE
                ]
                
                # Try different expiry formats
                expiry_formats = [
                    "25D12",    # 25Dec (D=December)
                    "251212",   # 2025-12-12
                    "D12",      # Dec 12
                    "1225",     # Dec 25
                ]
                
                print(f"\n🧪 Testing option symbol formats...")
                
                for strike in test_strikes:
                    print(f"\n--- Testing strike {strike} ---")
                    
                    for expiry in expiry_formats:
                        for fmt in option_formats:
                            try:
                                symbol = fmt.format(
                                    expiry=expiry,
                                    strike=int(strike),
                                    month="D",  # December
                                    day="12",   # 12th
                                    year="25"   # 2025
                                )
                                
                                print(f"  Testing: {symbol}")
                                
                                params = {'symbols': symbol}
                                response = requests.get(url, headers=headers, params=params, timeout=5)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    if data.get('s') == 'ok' and data.get('d'):
                                        quote_data = data['d'][0]
                                        if quote_data.get('s') == 'ok':
                                            price = quote_data['v'].get('lp', 0)
                                            print(f"    ✅ SUCCESS! Price: {price}")
                                            return symbol  # Return first working format
                                        else:
                                            print(f"    ❌ No data: {quote_data.get('message', 'Unknown')}")
                                    else:
                                        print(f"    ❌ API Error: {data.get('message', 'Unknown')}")
                                else:
                                    print(f"    ❌ HTTP {response.status_code}")
                                    
                            except Exception as e:
                                print(f"    ❌ Exception: {e}")
                
                print(f"\n❌ No working option symbol format found")
                return None
                
    except Exception as e:
        print(f"❌ Error getting NIFTY price: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Checking NIFTY Price and Option Symbol Formats")
    print("="*60)
    
    working_format = check_nifty_and_options()
    
    if working_format:
        print(f"\n✅ Found working option format: {working_format}")
    else:
        print(f"\n❌ No working option symbol format found")
        
    print("="*60)