import yaml
import datetime
import os

cfg_path = os.path.join('config', 'config.yaml')
print('Reading', cfg_path)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
fy = cfg.get('fyers', {})
print('client_id:', fy.get('client_id'))
print('access_token present:', bool(fy.get('access_token')))
print('token_expiry:', fy.get('token_expiry'))
exp = fy.get('token_expiry')
if exp:
    try:
        dt = datetime.datetime.strptime(exp, '%Y-%m-%d %H:%M:%S')
        now = datetime.datetime.now()
        print('expiry > now:', dt > now)
        print('seconds to expiry:', (dt - now).total_seconds())
    except Exception as e:
        print('expiry parse error', e)
else:
    print('no token_expiry set')
print('totp_key present:', bool(fy.get('totp_key')))
print('static_ips:', fy.get('static_ips') or fy.get('whitelisted_ips') or fy.get('static_ip'))
print('\nPaper trading default in strategy: strategy sets paper_trading=True in __init__')
