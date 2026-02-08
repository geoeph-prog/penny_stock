"""Reddit scraper using public JSON endpoints. No API key required."""

import re
import time

import requests
from loguru import logger

from pennystock.config import REDDIT_SUBREDDITS, REDDIT_POSTS_PER_SUB, REDDIT_DELAY_SECONDS


HEADERS = {
    "User-Agent": "PennyStockAnalyzer/1.0 (research tool)",
}


def search_ticker_in_subreddit(ticker: str, subreddit: str, limit: int = None) -> list:
    """
    Search a subreddit for mentions of a stock ticker.
    Returns list of dicts: {title, text, score, num_comments, created}.
    """
    limit = limit or REDDIT_POSTS_PER_SUB
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": ticker,
        "limit": min(limit, 100),
        "sort": "new",
        "restrict_sr": "on",
        "t": "week",
    }

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code == 429:
            logger.warning(f"Reddit rate limited on r/{subreddit}")
            return []
        if resp.status_code != 200:
            return []

        data = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")

            # Verify the ticker is actually mentioned (not just a substring match)
            combined = f"{title} {selftext}".upper()
            # Look for $TICKER or standalone TICKER (word boundary)
            if not re.search(rf'(\$|(?<!\w)){re.escape(ticker.upper())}(?!\w)', combined):
                continue

            posts.append({
                "title": title,
                "text": selftext[:500],  # Truncate long posts
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
                "created": post_data.get("created_utc", 0),
                "subreddit": subreddit,
            })

        return posts

    except requests.RequestException as e:
        logger.debug(f"Reddit request failed for {ticker} in r/{subreddit}: {e}")
        return []


def get_ticker_mentions(ticker: str, subreddits: list = None) -> dict:
    """
    Search all target subreddits for mentions of a ticker.

    Returns:
        {
            "ticker": str,
            "total_mentions": int,
            "total_score": int,
            "total_comments": int,
            "posts": list[dict],
            "subreddit_counts": dict[str, int],
        }
    """
    subreddits = subreddits or REDDIT_SUBREDDITS
    all_posts = []
    subreddit_counts = {}

    for sub in subreddits:
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


def get_subreddit_hot_posts(subreddit: str, limit: int = 50) -> list:
    """
    Get hot posts from a subreddit (for broad scanning).
    Returns list of post dicts.
    """
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    params = {"limit": min(limit, 100)}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            pd_ = child.get("data", {})
            posts.append({
                "title": pd_.get("title", ""),
                "text": pd_.get("selftext", "")[:500],
                "score": pd_.get("score", 0),
                "num_comments": pd_.get("num_comments", 0),
                "created": pd_.get("created_utc", 0),
                "subreddit": subreddit,
            })
        return posts

    except requests.RequestException:
        return []


def extract_tickers_from_posts(posts: list) -> dict:
    """
    Scan a list of posts for stock ticker mentions.
    Returns dict of {ticker: mention_count}.
    """
    ticker_pattern = re.compile(r'\$([A-Z]{2,5})\b')
    counts = {}

    for post in posts:
        text = f"{post.get('title', '')} {post.get('text', '')}"
        matches = ticker_pattern.findall(text)
        for match in matches:
            # Filter common false positives
            if match in {"USD", "CEO", "IPO", "ETF", "SEC", "FDA", "NYSE", "OTC", "ATH", "ATL", "EOD", "AMA", "PSA", "IMO", "FYI", "DD", "PT"}:
                continue
            counts[match] = counts.get(match, 0) + 1

    return counts
