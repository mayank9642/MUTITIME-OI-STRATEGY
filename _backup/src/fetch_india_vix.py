"""
Module to fetch India VIX data from NSE
India VIX is the volatility index for NIFTY 50
"""
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

NSE_VIX_URL = "https://www.nseindia.com/api/allIndices"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com"
}


def fetch_india_vix() -> Optional[float]:
    """
    Fetch current India VIX value from NSE
    
    Returns:
        float: Current India VIX value, or None if fetch fails
    """
    try:
        session = requests.Session()
        # Initial request to set cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        
        # Fetch all indices data
        response = session.get(NSE_VIX_URL, headers=NSE_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find India VIX in the data
        for index in data.get("data", []):
            if index.get("index") == "INDIA VIX":
                vix_value = float(index.get("last", 0))
                logger.info(f"India VIX fetched successfully: {vix_value}")
                return vix_value
        
        logger.warning("India VIX not found in NSE data")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching India VIX: {e}")
        return None
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing India VIX data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching India VIX: {e}")
        return None


def get_vix_market_condition(vix_value: float, min_threshold: float = 15.0, max_threshold: float = 35.0) -> dict:
    """
    Analyze market condition based on India VIX value
    
    Expert Classification of India VIX (Based on Market Standards):
    ================================================================
    
    HISTORICAL CONTEXT:
    - India VIX Average (2015-2023): ~17-18
    - Normal Range: 12-20
    - COVID-19 Peak (Mar 2020): 83+ (Extreme panic)
    - 2022 Russia-Ukraine: 30-35 (High fear)
    - Bull Market (2017): 10-12 (Complacency)
    
    EXPERT CLASSIFICATIONS:
    
    < 10:   EXTREME COMPLACENCY - Market overconfident, corrections likely
    10-15:  LOW VOLATILITY - Calm market, limited moves, choppy price action
    15-20:  NORMAL VOLATILITY - Healthy market, good for directional trades
    20-30:  ELEVATED VOLATILITY - Increased fear, strong trending moves
    30-40:  HIGH VOLATILITY - Significant uncertainty, sharp swings
    > 40:   EXTREME FEAR/PANIC - Crisis situation, avoid trading
    
    TRADING IMPLICATIONS:
    - VIX < 15: Sideways/Range-bound, False breakouts common
    - VIX 15-25: Ideal for breakout trading, sustained moves
    - VIX > 30: High risk, whipsaws, extreme uncertainty
    
    Args:
        vix_value (float): Current VIX value
        min_threshold (float): Minimum VIX to allow trading (default: 15.0)
        max_threshold (float): Maximum VIX to allow trading (default: 35.0)
        
    Returns:
        dict: Market condition analysis with should_trade flag
    """
    # Classification based on expert market analysis
    if vix_value < 10:
        return {
            "condition": "Extreme Complacency",
            "description": "Market overconfident. Correction/volatility spike likely soon.",
            "should_trade": False,
            "risk_level": "HIGH",
            "recommendation": f"Avoid trading - VIX {vix_value:.2f} indicates complacency (usually precedes sharp moves)"
        }
    elif vix_value < min_threshold:
        return {
            "condition": "Low Volatility (Choppy)",
            "description": "Calm market with limited directional moves. False breakouts common.",
            "should_trade": False,
            "risk_level": "HIGH",
            "recommendation": f"Avoid trading - VIX {vix_value:.2f} below threshold {min_threshold} (sideways market)"
        }
    elif vix_value < 20:
        return {
            "condition": "Normal Volatility",
            "description": "Healthy market conditions with good price discovery.",
            "should_trade": True,
            "risk_level": "LOW",
            "recommendation": "Good for trading - Normal trending conditions (VIX in sweet spot 15-20)"
        }
    elif vix_value < 25:
        return {
            "condition": "Elevated Volatility",
            "description": "Increased market uncertainty. Stronger directional moves expected.",
            "should_trade": True,
            "risk_level": "LOW",
            "recommendation": "Excellent for breakout trading - High momentum expected"
        }
    elif vix_value < 30:
        return {
            "condition": "High Volatility",
            "description": "Significant market fear. Sharp swings and quick reversals.",
            "should_trade": True,
            "risk_level": "MEDIUM",
            "recommendation": "Trade with caution - Use wider stops (VIX showing fear)"
        }
    elif vix_value < max_threshold:
        return {
            "condition": "Very High Volatility",
            "description": "Extreme uncertainty. Risk of violent whipsaws and gap moves.",
            "should_trade": True,
            "risk_level": "HIGH",
            "recommendation": f"High risk zone - Consider avoiding or reducing position size"
        }
    else:
        return {
            "condition": "Extreme Fear/Panic",
            "description": "Market crisis or panic selling. Extremely unpredictable.",
            "should_trade": False,
            "risk_level": "EXTREME",
            "recommendation": f"DO NOT TRADE - VIX {vix_value:.2f} indicates extreme panic (above {max_threshold})"
        }


if __name__ == "__main__":
    # Test the VIX fetching
    logging.basicConfig(level=logging.INFO)
    vix = fetch_india_vix()
    if vix:
        print(f"\nIndia VIX: {vix}")
        condition = get_vix_market_condition(vix)
        print(f"Market Condition: {condition['condition']}")
        print(f"Description: {condition['description']}")
        print(f"Should Trade: {condition['should_trade']}")
        print(f"Risk Level: {condition['risk_level']}")
        print(f"Recommendation: {condition['recommendation']}")
