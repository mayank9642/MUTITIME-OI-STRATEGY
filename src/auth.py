from fyers_apiv3 import fyersModel
import webbrowser
import urllib.parse
import json
import os
import sys
import yaml
import logging
import time
import datetime

# Ensure working directory is project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
os.chdir(project_root)

# Add the project root directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try using package import first (when running as module)
    from src.config import load_config
except ModuleNotFoundError:
    # Fall back to relative import (when running as script)
    from config import load_config

def generate_auth_code(use_totp=False):
    """
    Generate authentication code URL and open in browser
    
    Args:
        use_totp (bool): If True, use TOTP authentication instead of redirect URL
    """
    config = load_config()
    client_id = config['fyers']['client_id']
    redirect_uri = config['fyers']['redirect_uri']
    
    session = fyersModel.SessionModel(
        client_id=client_id,
        redirect_uri=redirect_uri,
        response_type="code",
        state="state"
    )
    
    auth_url = session.generate_authcode()
    print(f"Opening URL: {auth_url}")
    print("Please login and authorize the application...")
    webbrowser.open(auth_url, new=1)
    
    if use_totp and config.get('fyers', {}).get('totp_key'):
        try:
            import pyotp
            totp_key = config['fyers']['totp_key']
            totp = pyotp.TOTP(totp_key).now()
            print(f"Generated TOTP: {totp}")
            print("Using TOTP for authentication.")
            # When using TOTP, the flow is different and requires additional API calls
            # This is just a placeholder - implement according to Fyers TOTP flow
        except ImportError:
            print("pyotp package not installed. Please install it with: pip install pyotp")
            print("Falling back to URL redirect method.")
    
    # Get the auth code from the URL
    print("\nAfter authentication, you'll be redirected to a page.")
    print("Please copy the FULL URL from your browser's address bar and paste it below.")
    redirect_url = input("Enter the redirect URL: ")
    
    # Parse the auth code from the URL
    try:
        if "auth_code=" in redirect_url:
            auth_code = redirect_url[redirect_url.index('auth_code=')+10:redirect_url.index('&state')]
        else:
            print("Could not find auth_code in the URL. Please enter it manually.")
            auth_code = input("Enter the auth code: ")
        
        return auth_code
    except Exception as e:
        print(f"Error parsing auth code: {str(e)}")
        auth_code = input("Enter the auth code manually: ")
        return auth_code

def generate_access_token(use_totp=False):
    """
    Generate access token using the authorization code
    
    Args:
        use_totp (bool): If True, use TOTP authentication
        
    Returns:
        str: Access token if successful, None otherwise
    """
    try:
        config = load_config()
        client_id = config['fyers']['client_id']
        secret_key = config['fyers']['secret_key']
        redirect_uri = config['fyers']['redirect_uri']
        
        # Get authentication code
        auth_code = generate_auth_code(use_totp)
        
        # Create session model for token generation
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        
        # Set the auth code and generate token
        session.set_token(auth_code)
        response = session.generate_token()
        
        if response.get('access_token'):
            # Save the access token to config
            config['fyers']['access_token'] = response['access_token']
            
            # Save token expiry time (default is usually 1 day)
            expiry_time = datetime.datetime.now() + datetime.timedelta(days=1)
            config['fyers']['token_expiry'] = expiry_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write updated config to file
            with open('config/config.yaml', 'w') as f:
                yaml.dump(config, f)
                
            print(f"Access token generated and saved successfully.")
            print(f"Token valid until: {config['fyers']['token_expiry']}")
            
            # Also save token to access.txt for compatibility with example code
            with open('access.txt', 'w') as f:
                f.write(response['access_token'])
                
            return response['access_token']
        else:
            print(f"Failed to generate access token: {response}")
            return None
            
    except Exception as e:
        print(f"Error generating access token: {str(e)}")
        logging.error(f"Failed to generate access token: {str(e)}")
        return None

if __name__ == "__main__":
    token = generate_access_token()
    if token:
        print("Access token generated successfully!")
    else:
        print("Failed to generate access token.")