#!/usr/bin/env python
"""
Test script to verify that log filtering is working correctly.
This script generates sample logs with sensitive information and
verifies that they're properly filtered.
"""

import logging
import os
import re
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the filter from main
from src.main import SensitiveInfoFilter

# Configure a test log file
TEST_LOG_FILE = "logs/test_filtering.log"

def setup_test_logger():
    """Set up a test logger that writes to a file"""
    # Delete the test log file if it exists
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
    
    # Configure logging to write to file
    logging.basicConfig(
        filename=TEST_LOG_FILE,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Apply the sensitive info filter
    root_logger = logging.getLogger()
    root_logger.addFilter(SensitiveInfoFilter())
    
    return root_logger

def generate_test_logs():
    """Generate test logs with sensitive information"""
    logger = logging.getLogger()
    
    # Log some regular messages
    logger.info("This is a regular info message")
    logger.debug("This is a regular debug message")
    
    # Log some sensitive authentication information
    logger.debug("[DEBUG] get_fyers_client: client_id=ABCD1234, access_token=XYZ9876SENSITIVE")
    
    # Log option chain data structure
    logger.debug("Sample option data structure: {'field1': 'value1', 'field2': 'value2'}")
    
    # Log some more regular messages
    logger.info("Another regular message after sensitive logs")
    
    # Force log flush
    for handler in logger.handlers:
        handler.flush()

def verify_log_filtering():
    """Verify that sensitive information was properly filtered"""
    if not os.path.exists(TEST_LOG_FILE):
        print("ERROR: Test log file was not created")
        return False
    
    with open(TEST_LOG_FILE, 'r') as f:
        content = f.read()
    
    # Check that regular messages appear
    if "This is a regular info message" not in content:
        print("ERROR: Regular info message missing from log")
        return False
    
    if "This is a regular debug message" not in content:
        print("ERROR: Regular debug message missing from log")
        return False
    
    # Check that sensitive authentication info was filtered
    if re.search(r'client_id=ABCD1234', content):
        print("ERROR: Sensitive client_id was not filtered")
        return False
    
    if re.search(r'access_token=XYZ9876SENSITIVE', content):
        print("ERROR: Sensitive access_token was not filtered")
        return False
    
    # Check that option chain data structure was filtered
    if "Sample option data structure:" in content:
        print("ERROR: Option chain data structure was not filtered")
        return False
    
    print("SUCCESS: All log filtering tests passed")
    return True

def main():
    """Run the log filtering test"""
    print("Testing log filtering...")
    
    logger = setup_test_logger()
    generate_test_logs()
    result = verify_log_filtering()
    
    # Clean up the test log file
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
