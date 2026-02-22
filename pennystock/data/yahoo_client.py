"""Yahoo Finance data client with rate limiting and caching."""

import time
from io import StringIO

import pandas as pd
import yfinance as yf
from loguru import logger

from pennystock.config import (
    YAHOO_BATCH_SIZE,
    YAHOO_DELAY_SECONDS,
    YAHOO_MAX_RETRIES,
    HISTORY_PERIOD,
)
from pennystock.data.cache import cache_get, cache_set


def get_price_history(ticker: str, period: str = None) -> pd.DataFrame:
    """
    Get OHLCV price history for a single ticker.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    """
    period = period or HISTORY_PERIOD
    cache_key = f"price_{ticker}_{period}"
    cached = cache_get(cache_key)
    if cached is not None:
        try:
            return pd.read_json(StringIO(cached), orient="split")
        except Exception:
            pass

    for attempt in range(YAHOO_MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist is not None and not hist.empty:
                cache_set(cache_key, hist.to_json(orient="split", date_format="iso"))
                return hist
            return pd.DataFrame()
        except Exception as e:
            if attempt < YAHOO_MAX_RETRIES - 1:
                wait = YAHOO_DELAY_SECONDS * (2 ** attempt)
                logger.debug(f"Yahoo retry {attempt+1} for {ticker} after {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"Yahoo failed for {ticker} after {YAHOO_MAX_RETRIES} attempts: {e}")
                return pd.DataFrame()


def get_batch_history(tickers: list, period: str = None) -> dict:
    """
    Download price history for multiple tickers in batches.
    Returns dict of {ticker: DataFrame}.
    """
    period = period or HISTORY_PERIOD
    results = {}

    for i in range(0, len(tickers), YAHOO_BATCH_SIZE):
        batch = tickers[i:i + YAHOO_BATCH_SIZE]
        batch_str = " ".join(batch)

        try:
            data = yf.download(batch_str, period=period, group_by="ticker",
                               progress=False, threads=True)

            if len(batch) == 1:
                # yf.download returns flat DataFrame for single ticker
                if data is not None and not data.empty:
                    results[batch[0]] = data
            else:
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].dropna(how="all")
                            if not df.empty:
                                results[ticker] = df
                    except (KeyError, AttributeError):
                        pass
        except Exception as e:
            logger.warning(f"Batch download failed for {batch_str[:50]}...: {e}")

        if i + YAHOO_BATCH_SIZE < len(tickers):
            time.sleep(YAHOO_DELAY_SECONDS)

    logger.info(f"Yahoo batch: got data for {len(results)}/{len(tickers)} tickers")
    return results


def get_stock_info(ticker: str) -> dict:
    """
    Get fundamental data for a ticker: financials, balance sheet, info.
    Returns a dict with available fields.
    """
    cache_key = f"info_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ""),
            "short_name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "market_cap": info.get("marketCap", 0),
            "float_shares": info.get("floatShares", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "short_ratio": info.get("shortRatio", 0),
            "short_percent_of_float": info.get("shortPercentOfFloat", 0),
            "shares_short": info.get("sharesShort", 0),
            "shares_short_prior": info.get("sharesShortPriorMonth", 0),
            "insider_percent_held": info.get("heldPercentInsiders", 0),
            "institution_percent_held": info.get("heldPercentInstitutions", 0),
            "total_cash": info.get("totalCash", 0),
            "total_debt": info.get("totalDebt", 0),
            "total_revenue": info.get("totalRevenue", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
            "profit_margins": info.get("profitMargins", 0),
            "gross_margins": info.get("grossMargins", 0),
            "operating_margins": info.get("operatingMargins", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "current_ratio": info.get("currentRatio", 0),
            "book_value": info.get("bookValue", 0),
            "price_to_book": info.get("priceToBook", 0),
            "operating_cashflow": info.get("operatingCashflow", 0),
            "free_cashflow": info.get("freeCashflow", 0),
            "price": info.get("currentPrice", info.get("previousClose", 0)),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "avg_volume_10d": info.get("averageVolume10days", 0),
            "full_time_employees": info.get("fullTimeEmployees", 0),
            "audit_risk": info.get("auditRisk", 0),
            "overall_risk": info.get("overallRisk", 0),
        }

        cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Yahoo info failed for {ticker}: {e}")
        return {"ticker": ticker}


def get_news(ticker: str) -> list:
    """Get recent news articles for a ticker."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        articles = []
        for item in news[:15]:
            articles.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", 0),
            })
        return articles
    except Exception as e:
        logger.debug(f"Yahoo news failed for {ticker}: {e}")
        return []


def has_recent_reverse_split(ticker: str, months: int = 6) -> dict:
    """
    Check if a stock had a reverse split in the last N months.
    Reverse splits have ratio < 1.0 (e.g., 0.1 for 1-for-10).

    Returns:
        {"has_reverse_split": bool, "split_ratio": float or None, "split_date": str or None}
    """
    try:
        t = yf.Ticker(ticker)
        splits = t.splits
        if splits is None or splits.empty:
            return {"has_reverse_split": False, "split_ratio": None, "split_date": None}

        # Filter to recent splits
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=months * 30)

        for date, ratio in splits.items():
            split_date = pd.Timestamp(date)
            if split_date >= pd.Timestamp(cutoff) and ratio < 1.0:
                return {
                    "has_reverse_split": True,
                    "split_ratio": float(ratio),
                    "split_date": str(split_date.date()),
                }

        return {"has_reverse_split": False, "split_ratio": None, "split_date": None}

    except Exception as e:
        logger.debug(f"Reverse split check failed for {ticker}: {e}")
        return {"has_reverse_split": False, "split_ratio": None, "split_date": None}


def get_quarterly_financials(ticker: str) -> pd.DataFrame:
    """Get quarterly financials for revenue momentum analysis."""
    try:
        t = yf.Ticker(ticker)
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            return qf
        return pd.DataFrame()
    except Exception as e:
        logger.debug(f"Yahoo quarterly financials failed for {ticker}: {e}")
        return pd.DataFrame()
