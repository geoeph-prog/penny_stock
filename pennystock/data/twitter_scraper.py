"""
Optional Twitter/X scraper using twikit. Requires:
  1. pip install twikit
  2. A throwaway Twitter/X account (credentials in env vars or config)

Set TWITTER_ENABLED = True in config.py to activate.
Includes rate limit detection -- stops trying after first failure.
"""

import asyncio
import os
import time

from loguru import logger

from pennystock.config import TWITTER_ENABLED, TWITTER_COOKIES_FILE, TWITTER_TWEETS_PER_TICKER

# Global rate limit tracking
_rate_limited_until = 0


def _is_rate_limited():
    return time.time() < _rate_limited_until


def _set_rate_limited(seconds=300):
    global _rate_limited_until
    _rate_limited_until = time.time() + seconds
    logger.warning(f"Twitter rate limited. Pausing requests for {seconds}s.")


async def _search_cashtag(ticker: str, limit: int) -> list:
    """Internal async search for $TICKER on Twitter/X."""
    try:
        from twikit import Client
    except ImportError:
        logger.warning("twikit not installed. Run: pip install twikit")
        return []

    client = Client("en-US")

    # Try loading saved cookies first
    if os.path.exists(TWITTER_COOKIES_FILE):
        try:
            client.load_cookies(TWITTER_COOKIES_FILE)
        except Exception:
            logger.warning("Twitter cookies expired, need fresh login")
            return []
    else:
        # Need credentials for first login
        username = os.environ.get("TWITTER_USERNAME")
        email = os.environ.get("TWITTER_EMAIL")
        password = os.environ.get("TWITTER_PASSWORD")

        if not all([username, email, password]):
            logger.debug("Twitter credentials not set. Set TWITTER_USERNAME, TWITTER_EMAIL, TWITTER_PASSWORD env vars")
            return []

        try:
            await client.login(
                auth_info_1=username,
                auth_info_2=email,
                password=password,
            )
            client.save_cookies(TWITTER_COOKIES_FILE)
        except Exception as e:
            logger.warning(f"Twitter login failed: {e}")
            return []

    try:
        tweets = await client.search_tweet(f"${ticker}", "Latest")
        results = []
        count = 0
        for tweet in tweets:
            if count >= limit:
                break
            results.append({
                "text": tweet.text,
                "created": str(getattr(tweet, "created_at", "")),
                "likes": getattr(tweet, "favorite_count", 0),
                "retweets": getattr(tweet, "retweet_count", 0),
                "user": getattr(tweet, "user", {}).get("name", "") if isinstance(getattr(tweet, "user", None), dict) else "",
            })
            count += 1
        return results

    except Exception as e:
        logger.debug(f"Twitter search failed for ${ticker}: {e}")
        return []


def get_cashtag_tweets(ticker: str, limit: int = None) -> dict:
    """
    Search Twitter/X for $TICKER mentions.

    Returns:
        {
            "ticker": str,
            "tweets": list[dict],
            "total_count": int,
            "enabled": bool,
        }
    """
    if not TWITTER_ENABLED:
        return {"ticker": ticker, "tweets": [], "total_count": 0, "enabled": False}

    if _is_rate_limited():
        logger.debug(f"Twitter skipped for {ticker} (rate limited)")
        return {"ticker": ticker, "tweets": [], "total_count": 0, "enabled": True}

    limit = limit or TWITTER_TWEETS_PER_TICKER

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                tweets = pool.submit(
                    asyncio.run, _search_cashtag(ticker, limit)
                ).result(timeout=30)
        else:
            tweets = asyncio.run(_search_cashtag(ticker, limit))
    except Exception as e:
        logger.debug(f"Twitter scraping failed for {ticker}: {e}")
        # If it fails, assume rate limited and back off
        _set_rate_limited(300)
        tweets = []

    return {
        "ticker": ticker,
        "tweets": tweets,
        "total_count": len(tweets),
        "enabled": True,
    }
