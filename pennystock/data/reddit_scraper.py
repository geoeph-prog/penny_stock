"""
Reddit scraper using public JSON endpoints. No API key required.

Two modes:
  - Bulk mode: Download hot/new posts from all subreddits ONCE, search locally
  - Single mode: Search for one ticker (with backoff on rate limit)
"""

import re
import time

import requests
from loguru import logger

from pennystock.config import REDDIT_SUBREDDITS, REDDIT_POSTS_PER_SUB, REDDIT_DELAY_SECONDS


HEADERS = {
    "User-Agent": "PennyStockAnalyzer/2.0 (educational research; contact: research@example.com)",
}

# Global rate limit tracking
_rate_limited_until = 0
_bulk_cache = None  # Cached bulk posts: {subreddit: [posts]}
_bulk_cache_time = 0
BULK_CACHE_TTL = 1800  # 30 minutes


def _is_rate_limited():
    return time.time() < _rate_limited_until


def _set_rate_limited(seconds=60):
    global _rate_limited_until
    _rate_limited_until = time.time() + seconds
    logger.warning(f"Reddit rate limited. Pausing Reddit requests for {seconds}s.")


def _make_request(url, params, timeout=10):
    """Make a request with rate limit awareness and backoff."""
    if _is_rate_limited():
        return None

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)

        if resp.status_code == 429:
            # Exponential backoff: 60s, then 120s, then 300s
            wait = 60
            _set_rate_limited(wait)
            return None

        if resp.status_code != 200:
            return None

        return resp.json()

    except requests.RequestException as e:
        logger.debug(f"Reddit request failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# BULK MODE: Download once, search locally (preferred for many tickers)
# ════════════════════════════════════════════════════════════════════

def bulk_download_posts(subreddits: list = None, posts_per_sub: int = 100) -> dict:
    """
    Download recent posts from all target subreddits at once.
    Returns {subreddit: [post_dicts]}.
    """
    global _bulk_cache, _bulk_cache_time

    # Return cached if fresh
    if _bulk_cache and (time.time() - _bulk_cache_time) < BULK_CACHE_TTL:
        logger.debug("Using cached bulk Reddit posts")
        return _bulk_cache

    subreddits = subreddits or REDDIT_SUBREDDITS
    all_posts = {}

    logger.info(f"Bulk downloading Reddit posts from {len(subreddits)} subreddits...")

    for sub in subreddits:
        if _is_rate_limited():
            logger.debug(f"Skipping r/{sub} (rate limited)")
            all_posts[sub] = []
            continue

        posts = []

        # Get hot + new for better coverage (2 requests per sub)
        for sort in ["hot", "new"]:
            url = f"https://www.reddit.com/r/{sub}/{sort}.json"
            data = _make_request(url, {"limit": min(posts_per_sub, 100)})

            if data:
                for child in data.get("data", {}).get("children", []):
                    pd_ = child.get("data", {})
                    posts.append({
                        "title": pd_.get("title", ""),
                        "text": pd_.get("selftext", "")[:500],
                        "score": pd_.get("score", 0),
                        "num_comments": pd_.get("num_comments", 0),
                        "created": pd_.get("created_utc", 0),
                        "subreddit": sub,
                    })

            time.sleep(REDDIT_DELAY_SECONDS)

        all_posts[sub] = posts
        logger.debug(f"  r/{sub}: {len(posts)} posts")

    total = sum(len(p) for p in all_posts.values())
    logger.info(f"Bulk download complete: {total} posts from {len(subreddits)} subreddits")

    _bulk_cache = all_posts
    _bulk_cache_time = time.time()
    return all_posts


def search_bulk_posts(ticker: str, bulk_posts: dict = None) -> dict:
    """
    Search pre-downloaded bulk posts for a specific ticker.
    No API calls -- purely local string matching.
    """
    if bulk_posts is None:
        bulk_posts = _bulk_cache or {}

    ticker_upper = ticker.upper()
    pattern = re.compile(rf'(\$|(?<!\w)){re.escape(ticker_upper)}(?!\w)')

    all_matches = []
    subreddit_counts = {}

    for sub, posts in bulk_posts.items():
        matches = []
        for post in posts:
            combined = f"{post['title']} {post['text']}".upper()
            if pattern.search(combined):
                matches.append(post)

        subreddit_counts[sub] = len(matches)
        all_matches.extend(matches)

    return {
        "ticker": ticker,
        "total_mentions": len(all_matches),
        "total_score": sum(p["score"] for p in all_matches),
        "total_comments": sum(p["num_comments"] for p in all_matches),
        "posts": all_matches,
        "subreddit_counts": subreddit_counts,
    }


# ════════════════════════════════════════════════════════════════════
# SINGLE MODE: Search per ticker (fallback, used when only 1-2 tickers)
# ════════════════════════════════════════════════════════════════════

def search_ticker_in_subreddit(ticker: str, subreddit: str, limit: int = None) -> list:
    """Search a single subreddit for a ticker. Respects rate limits."""
    limit = limit or REDDIT_POSTS_PER_SUB

    if _is_rate_limited():
        return []

    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": ticker,
        "limit": min(limit, 100),
        "sort": "new",
        "restrict_sr": "on",
        "t": "week",
    }

    data = _make_request(url, params)
    if not data:
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        post_data = child.get("data", {})
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")

        combined = f"{title} {selftext}".upper()
        if not re.search(rf'(\$|(?<!\w)){re.escape(ticker.upper())}(?!\w)', combined):
            continue

        posts.append({
            "title": title,
            "text": selftext[:500],
            "score": post_data.get("score", 0),
            "num_comments": post_data.get("num_comments", 0),
            "created": post_data.get("created_utc", 0),
            "subreddit": subreddit,
        })

    return posts


def get_ticker_mentions(ticker: str, subreddits: list = None, use_bulk: bool = True) -> dict:
    """
    Get mentions of a ticker across all subreddits.

    Args:
        ticker: Stock ticker to search for.
        use_bulk: If True, uses bulk cached posts (fast, no extra API calls).
                  If False, makes individual search requests per subreddit.
    """
    # Prefer bulk mode (no API calls per ticker)
    if use_bulk and _bulk_cache:
        return search_bulk_posts(ticker)

    # Fallback to individual searches (but respect rate limits)
    if _is_rate_limited():
        return _empty_result(ticker)

    subreddits = subreddits or REDDIT_SUBREDDITS
    all_posts = []
    subreddit_counts = {}

    for sub in subreddits:
        if _is_rate_limited():
            subreddit_counts[sub] = 0
            continue

        posts = search_ticker_in_subreddit(ticker, sub)
        subreddit_counts[sub] = len(posts)
        all_posts.extend(posts)
        time.sleep(REDDIT_DELAY_SECONDS)

    return {
        "ticker": ticker,
        "total_mentions": len(all_posts),
        "total_score": sum(p["score"] for p in all_posts),
        "total_comments": sum(p["num_comments"] for p in all_posts),
        "posts": all_posts,
        "subreddit_counts": subreddit_counts,
    }


def _empty_result(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "total_mentions": 0,
        "total_score": 0,
        "total_comments": 0,
        "posts": [],
        "subreddit_counts": {},
    }


def extract_tickers_from_posts(posts: list) -> dict:
    """Scan posts for stock ticker mentions. Returns {ticker: count}."""
    ticker_pattern = re.compile(r'\$([A-Z]{2,5})\b')
    skip = {"USD", "CEO", "IPO", "ETF", "SEC", "FDA", "NYSE", "OTC",
            "ATH", "ATL", "EOD", "AMA", "PSA", "IMO", "FYI", "DD", "PT"}
    counts = {}

    for post in posts:
        text = f"{post.get('title', '')} {post.get('text', '')}"
        for match in ticker_pattern.findall(text):
            if match not in skip:
                counts[match] = counts.get(match, 0) + 1

    return counts
