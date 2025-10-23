"""
Setup script for OI-Strategy project
This script helps set up the project environment by creating necessary directories,
installing dependencies, and verifying the configuration.
"""

import os
import sys
import subprocess
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['logs', 'data']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
        else:
            logging.info(f"Directory already exists: {directory}")

def install_dependencies():
    """Install required Python packages"""
    try:
        logging.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing dependencies: {str(e)}")
        raise

def verify_config():
    """Verify that the config.yaml file has necessary fields"""
    try:
        config_path = "config/config.yaml"
        
        if not os.path.exists(config_path):
            logging.error(f"Config file not found: {config_path}")
            create_default_config()
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required_fields = {
            'fyers': ['client_id', 'secret_key', 'redirect_uri'],
            'strategy': ['symbol', 'analysis_time', 'max_holding_minutes', 'stoploss_pct', 'breakout_pct'],
            'logging': ['level', 'file']
        }
        
        missing_fields = []
        
        for section, fields in required_fields.items():
            if section not in config:
                missing_fields.append(section)
                continue
                
            for field in fields:
                if field not in config[section]:
                    missing_fields.append(f"{section}.{field}")
                    
        if missing_fields:
            logging.warning(f"Missing configuration fields: {', '.join(missing_fields)}")
            return False
        
        # Check for placeholder values
        if config['fyers']['client_id'] == "YOUR_FYERS_CLIENT_ID" or config['fyers']['secret_key'] == "YOUR_SECRET_KEY":
            logging.warning("Fyers API credentials contain placeholder values. Please update with actual values.")
            return False
            
        logging.info("Configuration verified successfully.")
        return True
        
    except Exception as e:
        logging.error(f"Error verifying configuration: {str(e)}")
        return False
        
def create_default_config():
    """Create a default config.yaml file"""
    try:
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        config_path = "config/config.yaml"
        
        default_config = {
            'fyers': {
                'client_id': "YOUR_FYERS_CLIENT_ID",
                'secret_key': "YOUR_SECRET_KEY",
                'redirect_uri': "https://fessorpro.com/",
                'access_token': "",
                'token_expiry': "",
                'totp_key': ""
            },
            'strategy': {
                'symbol': "NSE:NIFTY50-INDEX",
                'analysis_time': "09:20",
                'max_holding_minutes': 30,
                'risk_reward_ratio': 2,
                'stoploss_pct': 20,
                'breakout_pct': 10
            },
            'logging': {
                'level': "INFO",
                'file': "logs/strategy.log"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        logging.info(f"Created default configuration file: {config_path}")
        logging.info("Please update it with your actual Fyers API credentials.")
        
    except Exception as e:
        logging.error(f"Error creating default configuration: {str(e)}")

def main():
    """Main setup function"""
    logging.info("Setting up OI-Strategy project environment...")
    
    try:
        # Create necessary directories
        create_directories()
        
        # Install dependencies
        install_dependencies()
        
        # Verify configuration
        config_valid = verify_config()
        
        if config_valid:
            logging.info("Setup completed successfully!")
            logging.info("You can now run the main.py file to start the strategy.")
        else:
            logging.warning("Setup completed with warnings. Please review and fix the issues mentioned above.")
            
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        
if __name__ == "__main__":
    main()
