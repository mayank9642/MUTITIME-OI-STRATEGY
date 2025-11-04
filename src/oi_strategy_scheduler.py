"""
OI Strategy Scheduler - Runs OI strategy at scheduled time points
"""
import time
import logging
from datetime import datetime, timedelta
import sys, os
import pytz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategy import OpenInterestStrategy

IST = pytz.timezone('Asia/Kolkata')
SCHEDULED_TIMES = ["09:20", "10:30", "11:45", "12:30", "13:30", "14:30", "15:15"]

def get_next_run_time(now, times):
    today = now.date()
    for t in times:
        run_dt = IST.localize(datetime.strptime(f"{today} {t}", "%Y-%m-%d %H:%M"))
        logging.info(f"Checking scheduled time: {run_dt} (now: {now})")
        if run_dt > now:
            logging.info(f"Next run time selected: {run_dt}")
            return run_dt
    logging.info("No future scheduled times found.")
    return None

def main():
    logging.basicConfig(filename="logs/oi_scheduler.log", level=logging.INFO, filemode="w")
    logging.info("OI Scheduler started.")
    # Get current IST time
    now_ist = datetime.now(IST)
    # Wait for market open (09:15 IST)
    market_open = IST.localize(datetime.combine(now_ist.date(), datetime.strptime("09:15", "%H:%M").time()))
    while datetime.now(IST) < market_open:
        mins_left = int((market_open - datetime.now(IST)).total_seconds() // 60)
        logging.info(f"Waiting for market to open at 09:15 IST... {mins_left} min left")
        time.sleep(60)
    logging.info("Market opened at 09:15 IST.")
    times = SCHEDULED_TIMES.copy()
    while times:
        now = datetime.now(IST)
        next_run = get_next_run_time(now, times)
        if not next_run:
            logging.info("No more scheduled runs for today.")
            break
        wait_sec = (next_run - now).total_seconds()
        while wait_sec > 0:
            mins_left = int(wait_sec // 60)
            logging.info(f"Waiting for next run at {next_run.strftime('%H:%M')} IST... {mins_left} min left")
            sleep_time = min(60, wait_sec)
            time.sleep(sleep_time)
            wait_sec -= sleep_time
        logging.info(f"Running OI strategy at {next_run.strftime('%H:%M')} IST")
        strat = OpenInterestStrategy()
        strat.run_strategy()
        # Remove the time just used
        if next_run:
            times = [t for t in times if t != next_run.strftime('%H:%M')]
    logging.info("OI Scheduler finished for today.")

if __name__ == "__main__":
    main()
