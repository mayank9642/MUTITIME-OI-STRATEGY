"""Utility: convert option symbols to a compact canonical form.

Exposes:
- convert_option_symbol_format(symbol: Optional[str]) -> Optional[str]

The function is defensive: on parse failure it returns the original value.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _ensure_prefix(prefix: str, s: str) -> str:
    return s if s.startswith(prefix) else prefix + s


def convert_option_symbol_format(symbol: Optional[str]) -> Optional[str]:
    """Convert option symbol to compact canonical form or return original.

    Examples:
      - "NSE:NIFTY-04-Nov-2025-27450-CE" -> "NSE:NIFTY04NOV2527450CE"
      - "NIFTY_4_NOV_25_27450_CE" -> "NSE:NIFTY04NOV2527450CE"
    """
    if not symbol:
        return symbol

    try:
        # Fast path: already compact/canonical
        if re.match(r"^(NSE:)?NIFTY\d{2}[A-Z]{3}\d{2,4}\d{4,5}(CE|PE)$", symbol, re.IGNORECASE):
            return _ensure_prefix("NSE:", symbol)

        # If the token doesn't look like an option, leave as-is
        if "CE" not in symbol and "PE" not in symbol and not re.search(r"\b(CE|PE)\b", symbol, re.IGNORECASE):
            return symbol

        # Normalize separators for parsing
        s = symbol.replace("_", "-").replace(" ", "-")
        prefix = "NSE:"
        rest = s
        if ":" in s:
            parts = s.split(":", 1)
            prefix = parts[0] + ":"
            rest = parts[1]

        components = [c for c in rest.split("-") if c]

        # Expecting: UNDERLYING - DAY - MON - YEAR - STRIKE - TYPE
        if len(components) >= 6:
            underlying = components[0].upper()
            day = components[1].zfill(2)
            month = components[2].upper()[:3]
            year = components[3]
            strike = components[4].zfill(5)
            option_type = components[5].upper()
            if len(year) == 4:
                year = year[2:]
            expiry = f"{day}{month}{year}"
            return f"{prefix}{underlying}{expiry}{strike}{option_type}"

        # Fallback: regex extraction from messy strings
        m = re.search(
            r"NIFTY[-_]?(\d{1,2})[-_]?([A-Z]{3})[-_]?((?:\d{2})|(?:\d{4}))[-_]?(\d{4,5})[-_]?([CP]E)",
            rest,
            re.IGNORECASE,
        )
        if m:
            day, month, year, strike, option_type = m.groups()
            underlying = 'NIFTY'
            day = day.zfill(2)
            month = month.upper()[:3]
            if len(year) == 4:
                year = year[2:]
            strike = strike.zfill(5)
            option_type = option_type.upper()
            expiry = f"{day}{month}{year}"
            return f"{prefix}{underlying.upper()}{expiry}{strike}{option_type}"

        return symbol

    except Exception:
        logger.exception("Error converting option symbol: %s", symbol)
        return symbol
