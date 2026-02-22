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
    return await _search_cashtag_query(f"${ticker}", limit)


async def _search_cashtag_query(query: str, limit: int) -> list:
    """Internal async search for an arbitrary query on Twitter/X."""
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
        tweets = await client.search_tweet(query, "Latest")
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
        logger.debug(f"Twitter search failed for '{query}': {e}")
        return []


def get_cashtag_tweets(ticker: str, limit: int = None, aliases: list = None) -> dict:
    """
    Search Twitter/X for $TICKER mentions and company name aliases.

    Args:
        ticker: Stock ticker symbol.
        limit: Max tweets to fetch per search query.
        aliases: Company name aliases (e.g. ['Getty Images']) to search alongside $TICKER.

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

    # Build search queries: cashtag + company name aliases
    queries = [f"${ticker}"]
    for alias in (aliases or []):
        queries.append(f'"{alias}" stock')  # Add "stock" to reduce false positives

    all_tweets = []
    seen_texts = set()

    for query in queries:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    tweets = pool.submit(
                        asyncio.run, _search_cashtag_query(query, limit)
                    ).result(timeout=30)
            else:
                tweets = asyncio.run(_search_cashtag_query(query, limit))
        except Exception as e:
            logger.debug(f"Twitter scraping failed for query '{query}': {e}")
            _set_rate_limited(300)
            tweets = []

        # Deduplicate across queries
        for tw in tweets:
            text_key = tw["text"][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_tweets.append(tw)

    return {
        "ticker": ticker,
        "tweets": all_tweets,
        "total_count": len(all_tweets),
        "enabled": True,
    }
