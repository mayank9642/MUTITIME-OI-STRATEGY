from fyers_apiv3.FyersWebsocket import data_ws
import time
import yaml

# Read access token from file
with open('access.txt') as f:
    access_token = f.read().strip()

# Read client_id from config.yaml if available
try:
    with open('config/config.yaml') as ymlfile:
        config = yaml.safe_load(ymlfile)
        client_id = config['fyers']['client_id']
except Exception:
    client_id = 'YOUR_CLIENT_ID'  # Fallback, replace if needed

# Format: APP_ID:access_token
full_access_token = f"{client_id}:{access_token}"

# List your option symbols here (Fyers format)
symbols = [
    "NSE:NIFTY2580724800CE",
    "NSE:NIFTY2580724600PE"
]

def onmessage(msg):
    print(f"Custom: {msg}")

def onerror(msg):
    print(f"Error: {msg}")

def onclose(msg):
    print(f"Connection closed: {msg}")

def onopen():
    data_type = "SymbolUpdate"
    fyers_socket.subscribe(symbols=symbols, data_type=data_type)
    fyers_socket.keep_running()
    print('WebSocket subscription started')

fyers_socket = data_ws.FyersDataSocket(
    access_token=full_access_token,
    log_path="",
    litemode=False,
    write_to_file=False,
    reconnect=True,
    on_connect=onopen,
    on_close=onclose,
    on_error=onerror,
    on_message=onmessage
)

fyers_socket.connect()

# Keep the script running for a while to receive ticks
time.sleep(60)
