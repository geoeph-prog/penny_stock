"""Stock discovery via Finviz screener. No API key required."""

from loguru import logger

from pennystock.config import MIN_PRICE, MAX_PRICE, MIN_VOLUME


def get_penny_stocks(min_price=None, max_price=None, min_volume=None):
    """
    Screen Finviz for stocks in our price range ($0.50-$5.00).

    Returns list of dicts with at least: ticker, price, volume, company, sector.
    Falls back to an empty list on failure (caller should handle fallback).
    """
    min_price = min_price or MIN_PRICE
    max_price = max_price or MAX_PRICE
    min_volume = min_volume or MIN_VOLUME

    try:
        from finvizfinance.screener.overview import Overview

        foverview = Overview()

        # Finviz filter keys
        filters = {
            "Price": "Under $5",
            "Average Volume": "Over 50K",
        }
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()

        if df is None or df.empty:
            logger.warning("Finviz returned no results")
            return []

        # Normalize column names (finvizfinance uses title case)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Apply exact price range filter (Finviz "Under $5" is broad)
        if "price" in df.columns:
            df["price"] = df["price"].astype(float)
            df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]

        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(float)
            df = df[df["volume"] >= min_volume]

        stocks = []
        for _, row in df.iterrows():
            stocks.append({
                "ticker": row.get("ticker", ""),
                "company": row.get("company", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "price": float(row.get("price", 0)),
                "volume": float(row.get("volume", 0)),
                "change": float(row.get("change", "0").strip("%")) if isinstance(row.get("change"), str) else float(row.get("change", 0)),
                "market_cap": row.get("market_cap", ""),
            })

        logger.info(f"Finviz found {len(stocks)} stocks (${min_price}-${max_price}, vol>{min_volume:,})")
        return stocks

    except ImportError:
        logger.error("finvizfinance not installed. Run: pip install finvizfinance")
        return []
    except Exception as e:
        logger.error(f"Finviz screening failed: {e}")
        return []


def get_high_gainers(months=6, min_gain_pct=100):
    """
    Find stocks that gained significantly over the past N months.
    Used for pattern learning (analyzing what winners look like).
    """
    try:
        from finvizfinance.screener.overview import Overview

        foverview = Overview()

        # Map months to Finviz performance filter
        perf_map = {
            1: "Month +20%",
            3: "Quarter +50%",
            6: "Half +100%",
            12: "Year +100%",
        }

        perf_filter = perf_map.get(months, "Half +100%")

        filters = {
            "Price": "Under $5",
            "Average Volume": "Over 50K",
            "Performance": perf_filter,
        }
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()

        if df is None or df.empty:
            logger.warning("Finviz found no high gainers")
            return []

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        tickers = df["ticker"].tolist() if "ticker" in df.columns else []
        logger.info(f"Finviz found {len(tickers)} stocks with >{min_gain_pct}% gain over {months} months")
        return tickers

    except Exception as e:
        logger.error(f"Finviz high gainer scan failed: {e}")
        return []
