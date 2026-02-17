"""StockTwits API client. Free public API, no key required.

Includes rate limit detection and backoff to avoid hammering the API.
"""

import time

import requests
from loguru import logger

from pennystock.config import STOCKTWITS_URL

# Global rate limit tracking
_rate_limited_until = 0


def _is_rate_limited():
    return time.time() < _rate_limited_until


def _set_rate_limited(seconds=60):
    global _rate_limited_until
    _rate_limited_until = time.time() + seconds
    logger.warning(f"StockTwits rate limited. Pausing requests for {seconds}s.")


def get_messages(ticker: str) -> dict:
    """
    Get recent StockTwits messages for a ticker.
    Respects rate limits -- returns empty result if rate limited.

    Returns:
        {
            "ticker": str,
            "messages": list[dict],  # {body, sentiment, created}
            "bullish_count": int,
            "bearish_count": int,
            "total_count": int,
            "rate_limited": bool,
        }
    """
    if _is_rate_limited():
        logger.debug(f"StockTwits skipped for {ticker} (rate limited)")
        result = _empty_result(ticker)
        result["rate_limited"] = True
        return result

    url = STOCKTWITS_URL.format(ticker=ticker.upper())

    try:
        resp = requests.get(url, timeout=10)

        if resp.status_code == 429:
            _set_rate_limited(120)
            result = _empty_result(ticker)
            result["rate_limited"] = True
            return result

        if resp.status_code != 200:
            logger.debug(f"StockTwits returned {resp.status_code} for {ticker}")
            return _empty_result(ticker)

        data = resp.json()
        messages_data = data.get("messages", [])

        messages = []
        bullish = 0
        bearish = 0

        for msg in messages_data:
            sentiment_obj = msg.get("entities", {}).get("sentiment")
            sentiment = None
            if sentiment_obj:
                sentiment = sentiment_obj.get("basic")
                if sentiment == "Bullish":
                    bullish += 1
                elif sentiment == "Bearish":
                    bearish += 1

            messages.append({
                "body": msg.get("body", ""),
                "sentiment": sentiment,
                "created": msg.get("created_at", ""),
            })

        return {
            "ticker": ticker,
            "messages": messages,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "total_count": len(messages),
            "rate_limited": False,
        }

    except requests.RequestException as e:
        logger.debug(f"StockTwits failed for {ticker}: {e}")
        return _empty_result(ticker)


def _empty_result(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "messages": [],
        "bullish_count": 0,
        "bearish_count": 0,
        "total_count": 0,
        "rate_limited": False,
    }
