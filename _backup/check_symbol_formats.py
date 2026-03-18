import sys
sys.path.append('c:/vs code projects/MUTITIME-OI-STRATEGY')
from src.nse_data_new import get_nifty_option_chain

print("Checking Fyers symbol format across multiple expiries:\n")
for i in [0, 1, 2]:
    df = get_nifty_option_chain(i)
    if not df.empty:
        # Get a 26000 CE strike to see the symbol format
        sample = df[(df['strikePrice'] == 26000) & (df['option_type'] == 'CE')]
        if not sample.empty:
            symbol = sample['symbol'].iloc[0]
            expiry = sample['expiry'].iloc[0]
            print(f"Expiry {i}: {expiry} -> {symbol}")
        else:
            print(f"Expiry {i}: No 26000 CE strike found")
    else:
        print(f"Expiry {i}: No data")
