"""Crypto ticker/entity extraction from social text.

Two signals: cashtags ($BTC) and an alias map (bitcoin -> BTC). Aliases are
deliberately conservative — ambiguous English words ('link', 'dot', 'ada')
only match their unambiguous project names.
"""

from __future__ import annotations

import re

TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "bitcoin"),
    "ETH": ("eth", "ethereum"),
    "SOL": ("sol", "solana"),
    "XRP": ("xrp", "ripple"),
    "DOGE": ("doge", "dogecoin"),
    "ADA": ("cardano",),
    "AVAX": ("avax", "avalanche"),
    "LINK": ("chainlink",),
    "DOT": ("polkadot",),
    "BNB": ("bnb", "binance coin"),
    "MATIC": ("matic", "polygon"),
    "LTC": ("ltc", "litecoin"),
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{2,10})\b")
_ALIAS_RES: dict[str, re.Pattern[str]] = {
    ticker: re.compile(
        r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE
    )
    for ticker, aliases in TICKER_ALIASES.items()
}


def extract_tickers(text: str) -> list[str]:
    """Return sorted, deduplicated ticker symbols mentioned in text."""
    found: set[str] = set()
    for match in _CASHTAG_RE.finditer(text):
        symbol = match.group(1).upper()
        if symbol in TICKER_ALIASES:
            found.add(symbol)
    for ticker, pattern in _ALIAS_RES.items():
        if pattern.search(text):
            found.add(ticker)
    return sorted(found)
