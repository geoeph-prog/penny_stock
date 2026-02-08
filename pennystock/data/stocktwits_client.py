"""StockTwits API client. Free public API, no key required."""

import requests
from loguru import logger

from pennystock.config import STOCKTWITS_URL


def get_messages(ticker: str) -> dict:
    """
    Get recent StockTwits messages for a ticker.

    Returns:
        {
            "ticker": str,
            "messages": list[dict],  # {body, sentiment, created}
            "bullish_count": int,
            "bearish_count": int,
            "total_count": int,
        }
    """
    url = STOCKTWITS_URL.format(ticker=ticker.upper())

    try:
        resp = requests.get(url, timeout=10)
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
    }
