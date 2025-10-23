import logging

# Set up logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

# Define the filter
class SensitiveInfoFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        
        if message.startswith('[DEBUG] get_fyers_client:'):
            record.msg = '[DEBUG] get_fyers_client: <CREDENTIALS FILTERED>'
            return True
            
        if 'Sample option data structure:' in message:
            return False
            
        return True

# Apply the filter
logging.getLogger().addFilter(SensitiveInfoFilter())

# Test the filter
print("Testing log filtering...")
print("------------------------")

print("\nTest 1: Regular message (should appear):")
logging.info('Regular message')

print("\nTest 2: Authentication token (should be filtered):")
logging.debug('[DEBUG] get_fyers_client: token=SECRET')

print("\nTest 3: Option data structure (should be dropped, nothing should appear below this line):")
logging.debug('Sample option data structure: {"data": "test"}')

print("\nDone testing")
