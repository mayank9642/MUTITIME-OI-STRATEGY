"""
Utilities for checking market status and handling market-closed scenarios
"""
import datetime
import pytz
import logging

def is_market_open():
    """
    Check if the market is currently open based on time and day of week.
    Returns a tuple (is_open, message) where is_open is a boolean and message is a descriptive string.
    """
    # Get current time in IST
    ist_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist_tz)
    
    # Check if it's a weekday (0=Monday, 6=Sunday)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False, f"Market closed - weekend ({['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][now.weekday()]})"
    
    # Define market hours (9:15 AM to 3:30 PM IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Check if current time is within market hours
    if now < market_open:
        minutes_to_open = (market_open - now).total_seconds() / 60
        return False, f"Market not yet open. Opens in {int(minutes_to_open)} minutes"
    elif now > market_close:
        return False, f"Market closed at 3:30 PM IST"
    
    return True, "Market is open"

def check_and_log_market_status():
    """
    Check if market is open and log the status.
    Returns True if market is open, False otherwise.
    """
    is_open, status_message = is_market_open()
    log_level = logging.INFO if is_open else logging.WARNING
    logging.log(log_level, f"Market status: {status_message}")
    return is_open

def get_time_to_market_open():
    """
    Get the time remaining until market open.
    Returns a tuple (seconds_to_open, formatted_time_string)
    """
    # Get current time in IST
    ist_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist_tz)
    
    # Define market open time
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    # If market open is in the past for today, calculate for tomorrow
    if now > market_open:
        market_open = market_open + datetime.timedelta(days=1)
        
        # If tomorrow is weekend, adjust to Monday
        weekday = market_open.weekday()
        if weekday >= 5:  # Saturday or Sunday
            days_to_add = 7 - weekday  # Add days to reach Monday
            market_open = market_open + datetime.timedelta(days=days_to_add)
    
    # Calculate time difference
    time_diff = (market_open - now).total_seconds()
    
    # Format as human-readable string
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        formatted_time = f"{int(minutes)}m {int(seconds)}s"
    
    return time_diff, formatted_time
