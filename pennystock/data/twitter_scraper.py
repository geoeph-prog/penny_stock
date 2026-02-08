"""
Optional Twitter/X scraper using twikit. Requires:
  1. pip install twikit
  2. A throwaway Twitter/X account (credentials in env vars or config)

Set TWITTER_ENABLED = True in config.py to activate.
"""

import asyncio
import os

from loguru import logger

from pennystock.config import TWITTER_ENABLED, TWITTER_COOKIES_FILE, TWITTER_TWEETS_PER_TICKER


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

    limit = limit or TWITTER_TWEETS_PER_TICKER

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                tweets = pool.submit(
                    asyncio.run, _search_cashtag(ticker, limit)
                ).result(timeout=30)
        else:
            tweets = asyncio.run(_search_cashtag(ticker, limit))
    except Exception as e:
        logger.debug(f"Twitter scraping failed for {ticker}: {e}")
        tweets = []

    return {
        "ticker": ticker,
        "tweets": tweets,
        "total_count": len(tweets),
        "enabled": True,
    }
