import requests
import yaml

# Minimal Fyers API option chain test
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
access_token = config['fyers']['access_token']
client_id = config['fyers']['client_id']

url = "https://api.fyers.in/api/v3/option_chain"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}
payload = {
    "symbol": "NSE:NIFTY50-INDEX",
    "expiry": "",
    "strikeCount": 5
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
