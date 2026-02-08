"""News catalyst detection using Yahoo Finance news and VADER NLP."""

from loguru import logger

from pennystock.config import POSITIVE_CATALYSTS, NEGATIVE_CATALYSTS
from pennystock.data.yahoo_client import get_news


def analyze(ticker: str) -> dict:
    """
    Scan recent news for catalysts.

    Returns:
        {
            "score": float (0-100),
            "positive_catalysts": list[str],
            "negative_catalysts": list[str],
            "total_articles": int,
            "catalyst_density": float,  # catalysts per article
        }
    """
    articles = get_news(ticker)

    if not articles:
        return {
            "score": 50,  # Neutral -- no news could be good or bad
            "positive_catalysts": [],
            "negative_catalysts": [],
            "total_articles": 0,
            "catalyst_density": 0,
        }

    positive_found = []
    negative_found = []

    for article in articles:
        title = article.get("title", "").lower()

        for keyword in POSITIVE_CATALYSTS:
            if keyword in title:
                positive_found.append(f"{keyword}: {article['title']}")
                break  # Count each article once

        for keyword in NEGATIVE_CATALYSTS:
            if keyword in title:
                negative_found.append(f"{keyword}: {article['title']}")
                break

    # Score based on catalyst balance
    pos_count = len(positive_found)
    neg_count = len(negative_found)
    total = len(articles)

    score = 50  # Baseline

    # Positive catalysts boost score
    score += min(30, pos_count * 10)

    # Negative catalysts reduce score
    score -= min(30, neg_count * 12)

    # Having recent news at all is slightly positive (attention)
    if total > 5:
        score += 5
    elif total > 0:
        score += 2

    # Net catalyst sentiment
    if pos_count > 0 and neg_count == 0:
        score += 10  # All positive news
    elif neg_count > 0 and pos_count == 0:
        score -= 10  # All negative news

    density = (pos_count + neg_count) / total if total > 0 else 0

    return {
        "score": round(max(0, min(100, score)), 1),
        "positive_catalysts": positive_found,
        "negative_catalysts": negative_found,
        "total_articles": total,
        "catalyst_density": round(density, 2),
    }
