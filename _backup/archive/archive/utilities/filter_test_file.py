import logging
import os

# Set up file for output
output_file = "filter_test_output.txt"
with open(output_file, 'w') as f:
    f.write("Test started\n")

# Set up logging to write to the file
logging.basicConfig(
    filename=output_file,
    level=logging.DEBUG,
    format='%(message)s'
)

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

# Write to the file directly for clarity
def write_to_file(message):
    with open(output_file, 'a') as f:
        f.write(message + "\n")

# Test the filter
write_to_file("\nTesting log filtering...")
write_to_file("------------------------")

write_to_file("\nTest 1: Regular message (should appear):")
logging.info('Regular message')

write_to_file("\nTest 2: Authentication token (should be filtered):")
logging.debug('[DEBUG] get_fyers_client: token=SECRET')

write_to_file("\nTest 3: Option data structure (should be dropped, nothing should appear between these lines):")
logging.debug('Sample option data structure: {"data": "test"}')
write_to_file("End of Test 3")

write_to_file("\nDone testing")

# Print the location of the output file
print(f"Test results written to: {os.path.abspath(output_file)}")
