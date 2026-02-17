"""
Sentiment analysis engine using VADER NLP.

Aggregates text from Reddit, StockTwits, and optionally Twitter,
then scores sentiment using VADER with financial term adjustments.

Supports bulk mode: call ensure_bulk_downloaded() once before analyzing
many tickers to avoid per-ticker Reddit API calls.
"""

from loguru import logger

from pennystock.config import FINANCIAL_LEXICON_UPDATES
from pennystock.data import reddit_scraper, stocktwits_client, twitter_scraper


def ensure_bulk_downloaded():
    """
    Pre-download Reddit posts from all subreddits in one batch.
    Call this ONCE before analyzing many tickers to avoid rate limiting.
    After this, reddit_scraper.get_ticker_mentions() uses local search only.
    """
    reddit_scraper.bulk_download_posts()
    logger.info("Reddit bulk download ready — ticker searches will be local only.")


def _get_vader():
    """Lazy-load and configure VADER with financial lexicon."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        logger.error("vaderSentiment not installed. Run: pip install vaderSentiment")
        return None

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCIAL_LEXICON_UPDATES)
    return analyzer


def score_text(text: str, analyzer=None) -> float:
    """
    Score a single text string for sentiment.
    Returns float in [-1, 1] where -1 = very negative, 1 = very positive.
    """
    if not text or not text.strip():
        return 0.0

    if analyzer is None:
        analyzer = _get_vader()
    if analyzer is None:
        return 0.0

    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def score_texts(texts: list, analyzer=None) -> dict:
    """
    Score a list of texts and return aggregate metrics.

    Returns:
        {
            "avg_sentiment": float,    # Average compound score [-1, 1]
            "positive_pct": float,     # % of texts with positive sentiment
            "negative_pct": float,     # % of texts with negative sentiment
            "neutral_pct": float,      # % of texts with neutral sentiment
            "count": int,
        }
    """
    if not texts:
        return {"avg_sentiment": 0, "positive_pct": 0, "negative_pct": 0,
                "neutral_pct": 100, "count": 0}

    if analyzer is None:
        analyzer = _get_vader()
    if analyzer is None:
        return {"avg_sentiment": 0, "positive_pct": 0, "negative_pct": 0,
                "neutral_pct": 100, "count": 0}

    scores = [score_text(t, analyzer) for t in texts]
    positive = sum(1 for s in scores if s > 0.05)
    negative = sum(1 for s in scores if s < -0.05)
    neutral = len(scores) - positive - negative

    return {
        "avg_sentiment": sum(scores) / len(scores),
        "positive_pct": positive / len(scores) * 100,
        "negative_pct": negative / len(scores) * 100,
        "neutral_pct": neutral / len(scores) * 100,
        "count": len(scores),
    }


def analyze(ticker: str) -> dict:
    """
    Run full sentiment analysis for a ticker across all sources.

    Returns:
        {
            "score": float (0-100),
            "reddit": dict,
            "stocktwits": dict,
            "twitter": dict,
            "combined_sentiment": float (-1 to 1),
            "total_mentions": int,
            "buzz_score": float (0-100),
        }
    """
    analyzer = _get_vader()

    # ── Reddit (uses bulk cache if available, zero API calls) ──────
    reddit_data = reddit_scraper.get_ticker_mentions(ticker, use_bulk=True)
    reddit_texts = [f"{p['title']} {p['text']}" for p in reddit_data["posts"]]
    reddit_sentiment = score_texts(reddit_texts, analyzer)

    # ── StockTwits (respects rate limits) ───────────────────────────
    st_data = stocktwits_client.get_messages(ticker)
    st_texts = [m["body"] for m in st_data["messages"]]
    st_sentiment = score_texts(st_texts, analyzer)

    # StockTwits also has its own sentiment labels
    st_total = st_data["total_count"] or 1
    st_label_score = (st_data["bullish_count"] - st_data["bearish_count"]) / st_total

    # ── Twitter (optional, respects rate limits) ────────────────────
    tw_data = twitter_scraper.get_cashtag_tweets(ticker)
    tw_texts = [t["text"] for t in tw_data["tweets"]]
    tw_sentiment = score_texts(tw_texts, analyzer)

    # ── Combine sentiment scores ────────────────────────────────────
    # Weight by source reliability and coverage
    sources = []
    weights = []

    if reddit_sentiment["count"] > 0:
        sources.append(reddit_sentiment["avg_sentiment"])
        weights.append(0.35)
    if st_sentiment["count"] > 0:
        # Blend VADER score with StockTwits native labels
        combined_st = st_sentiment["avg_sentiment"] * 0.5 + st_label_score * 0.5
        sources.append(combined_st)
        weights.append(0.35)
    if tw_sentiment["count"] > 0:
        sources.append(tw_sentiment["avg_sentiment"])
        weights.append(0.30)

    if sources:
        total_weight = sum(weights)
        combined = sum(s * w for s, w in zip(sources, weights)) / total_weight
    else:
        combined = 0.0

    total_mentions = (
        reddit_data["total_mentions"] +
        st_data["total_count"] +
        tw_data["total_count"]
    )

    # ── Buzz score: how much attention is this stock getting? ──────
    # More mentions = more buzz, logarithmic scale
    import math
    if total_mentions > 0:
        buzz = min(100, math.log(total_mentions + 1) * 20)
    else:
        buzz = 0

    # ── Composite score (0-100) ─────────────────────────────────────
    # Convert combined sentiment [-1, 1] to [0, 100]
    sentiment_0_100 = (combined + 1) * 50

    # Blend sentiment with buzz (80% sentiment, 20% buzz)
    score = sentiment_0_100 * 0.8 + buzz * 0.2

    return {
        "score": round(max(0, min(100, score)), 1),
        "combined_sentiment": round(combined, 3),
        "total_mentions": total_mentions,
        "buzz_score": round(buzz, 1),
        "reddit": {
            "mentions": reddit_data["total_mentions"],
            "avg_sentiment": round(reddit_sentiment["avg_sentiment"], 3),
            "positive_pct": round(reddit_sentiment["positive_pct"], 1),
            "subreddit_counts": reddit_data["subreddit_counts"],
        },
        "stocktwits": {
            "total": st_data["total_count"],
            "bullish": st_data["bullish_count"],
            "bearish": st_data["bearish_count"],
            "avg_sentiment": round(st_sentiment["avg_sentiment"], 3),
        },
        "twitter": {
            "total": tw_data["total_count"],
            "avg_sentiment": round(tw_sentiment["avg_sentiment"], 3),
            "enabled": tw_data["enabled"],
        },
    }
