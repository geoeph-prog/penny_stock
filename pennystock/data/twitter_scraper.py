"""
Twitter/X scraper with multi-strategy approach.

Strategy 1 (default, no account needed):
  Scrapes DuckDuckGo for recent Twitter/X posts about a stock ticker.
  Slower but completely free and requires zero credentials.

Strategy 2 (optional, needs throwaway account):
  Uses twikit library to access Twitter's internal API.
  Set TWITTER_ENABLED = True and provide credentials to activate.

Both strategies include rate limit detection and backoff.
"""

import re
import time

import requests
from bs4 import BeautifulSoup
from loguru import logger

from pennystock.config import TWITTER_ENABLED, TWITTER_COOKIES_FILE, TWITTER_TWEETS_PER_TICKER

# Global rate limit tracking
_rate_limited_until = 0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _is_rate_limited():
    return time.time() < _rate_limited_until


def _set_rate_limited(seconds=300):
    global _rate_limited_until
    _rate_limited_until = time.time() + seconds
    logger.debug(f"Twitter/web search rate limited for {seconds}s.")


# ════════════════════════════════════════════════════════════════════
# STRATEGY 1: Web Search Scraping (no account needed)
# ════════════════════════════════════════════════════════════════════

def _scrape_web_search(query: str, max_results: int = 15) -> list:
    """
    Search DuckDuckGo HTML for Twitter/X posts about a topic.
    Returns list of tweet-like dicts with text snippets.
    """
    results = []

    # DuckDuckGo HTML endpoint — no JS required
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}

    try:
        resp = requests.post(url, data=params, headers=HEADERS, timeout=15)
        if resp.status_code == 429:
            _set_rate_limited(120)
            return []
        if resp.status_code != 200:
            logger.debug(f"DuckDuckGo returned {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", class_="result__a")
        snippets = soup.find_all("a", class_="result__snippet")

        for i, (link, snippet) in enumerate(zip(links, snippets)):
            if i >= max_results:
                break
            href = link.get("href", "")
            title = link.get_text(strip=True)
            text = snippet.get_text(strip=True) if snippet else title

            # Keep results that look like they're from X/Twitter
            if text and len(text) > 20:
                results.append({
                    "text": text,
                    "title": title,
                    "source_url": href,
                    "created": "",
                    "likes": 0,
                    "retweets": 0,
                    "user": "",
                })
    except requests.RequestException as e:
        logger.debug(f"Web search failed for '{query}': {e}")
    except Exception as e:
        logger.debug(f"Web search parse error: {e}")

    return results


def _search_twitter_via_web(ticker: str, aliases: list = None, limit: int = 15) -> list:
    """
    Search for Twitter/X mentions of a stock via web search engines.
    No Twitter account required.
    """
    all_results = []
    seen_texts = set()

    # Build search queries targeting Twitter/X content
    queries = [
        f'site:x.com "${ticker}" stock',
        f'site:twitter.com "${ticker}" stock',
    ]
    # Also search by company name aliases
    for alias in (aliases or []):
        queries.append(f'site:x.com "{alias}" stock')

    for query in queries:
        if _is_rate_limited():
            break
        results = _scrape_web_search(query, max_results=limit)
        for r in results:
            text_key = r["text"][:80]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(r)
        # Small delay between queries to be respectful
        time.sleep(1.0)

    return all_results[:limit]


# ════════════════════════════════════════════════════════════════════
# STRATEGY 2: twikit (requires throwaway account, optional)
# ════════════════════════════════════════════════════════════════════

def _search_twikit(ticker: str, aliases: list = None, limit: int = 30) -> list:
    """Use twikit if installed and configured. Returns tweet list or empty."""
    import asyncio
    import os

    try:
        from twikit import Client
    except ImportError:
        return []

    async def _do_search(query, lim):
        client = Client("en-US")
        if os.path.exists(TWITTER_COOKIES_FILE):
            try:
                client.load_cookies(TWITTER_COOKIES_FILE)
            except Exception:
                return []
        else:
            username = os.environ.get("TWITTER_USERNAME")
            email = os.environ.get("TWITTER_EMAIL")
            password = os.environ.get("TWITTER_PASSWORD")
            if not all([username, email, password]):
                return []
            try:
                await client.login(
                    auth_info_1=username, auth_info_2=email, password=password,
                )
                client.save_cookies(TWITTER_COOKIES_FILE)
            except Exception:
                return []

        try:
            tweets = await client.search_tweet(query, "Latest")
            out = []
            for i, tweet in enumerate(tweets):
                if i >= lim:
                    break
                out.append({
                    "text": tweet.text,
                    "created": str(getattr(tweet, "created_at", "")),
                    "likes": getattr(tweet, "favorite_count", 0),
                    "retweets": getattr(tweet, "retweet_count", 0),
                    "user": getattr(tweet, "user", {}).get("name", "")
                           if isinstance(getattr(tweet, "user", None), dict) else "",
                })
            return out
        except Exception:
            return []

    queries = [f"${ticker}"]
    for alias in (aliases or []):
        queries.append(f'"{alias}" stock')

    all_tweets = []
    seen = set()

    for q in queries:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    tweets = pool.submit(asyncio.run, _do_search(q, limit)).result(timeout=30)
            else:
                tweets = asyncio.run(_do_search(q, limit))
        except Exception:
            _set_rate_limited(300)
            tweets = []
        for tw in tweets:
            key = tw["text"][:80]
            if key not in seen:
                seen.add(key)
                all_tweets.append(tw)

    return all_tweets


# ════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════

def get_cashtag_tweets(ticker: str, limit: int = None, aliases: list = None) -> dict:
    """
    Search Twitter/X for $TICKER mentions and company name aliases.

    Strategy:
      1. If TWITTER_ENABLED and twikit configured → use twikit (best data)
      2. Otherwise → scrape web search for Twitter/X mentions (no auth)

    Args:
        ticker: Stock ticker symbol.
        limit: Max tweets to fetch per search query.
        aliases: Company name aliases (e.g. ['Getty Images']).

    Returns:
        {
            "ticker": str,
            "tweets": list[dict],
            "total_count": int,
            "enabled": bool,
            "source": str,  # "twikit", "web_search", or "disabled"
        }
    """
    if _is_rate_limited():
        logger.debug(f"Twitter skipped for {ticker} (rate limited)")
        return {"ticker": ticker, "tweets": [], "total_count": 0,
                "enabled": True, "source": "rate_limited"}

    limit = limit or TWITTER_TWEETS_PER_TICKER

    # Strategy 1: Try twikit if enabled and configured
    if TWITTER_ENABLED:
        tweets = _search_twikit(ticker, aliases=aliases, limit=limit)
        if tweets:
            return {
                "ticker": ticker,
                "tweets": tweets,
                "total_count": len(tweets),
                "enabled": True,
                "source": "twikit",
            }

    # Strategy 2: Web search scraping (always available, no auth needed)
    tweets = _search_twitter_via_web(ticker, aliases=aliases, limit=limit)
    return {
        "ticker": ticker,
        "tweets": tweets,
        "total_count": len(tweets),
        "enabled": True,
        "source": "web_search",
    }
