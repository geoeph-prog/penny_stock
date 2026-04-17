"""Stock discovery via Finviz screener ($2-$5 range). No API key required."""

from loguru import logger

from pennystock.config import MIN_PRICE, MAX_PRICE, MIN_VOLUME


def _normalize_columns(df):
    """Normalize DataFrame columns: lowercase, strip, replace spaces."""
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


def _to_float(val):
    """Coerce a value to float, stripping common suffixes like %, $, commas."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().replace(",", "").replace("$", "").replace("%", "")
        if not val or val == "-":
            return 0.0
        return float(val)
    return 0.0


def get_penny_stocks(min_price=None, max_price=None, min_volume=None):
    """
    Screen Finviz for stocks in our price range ($2-$5).

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
            "Average Volume": "Over 100K",
        }
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()

        if df is None or df.empty:
            logger.warning("Finviz returned no results")
            return []

        _normalize_columns(df)
        logger.debug(f"Finviz raw columns: {list(df.columns)}")
        logger.debug(f"Finviz raw rows: {len(df)}")

        if "price" in df.columns:
            df["price"] = df["price"].apply(_to_float)
            before = len(df)
            df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
            logger.debug(f"Price filter ${min_price}-${max_price}: {before} -> {len(df)}")
        else:
            logger.warning(f"No 'price' column found in Finviz data (columns: {list(df.columns)})")

        if "volume" in df.columns:
            df["volume"] = df["volume"].apply(_to_float)
            before = len(df)
            df = df[df["volume"] >= min_volume]
            logger.debug(f"Volume filter >={min_volume:,}: {before} -> {len(df)}")
        else:
            logger.warning(f"No 'volume' column found in Finviz data (columns: {list(df.columns)})")

        stocks = []
        for _, row in df.iterrows():
            change_raw = row.get("change", 0)
            stocks.append({
                "ticker": row.get("ticker", ""),
                "company": row.get("company", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "price": _to_float(row.get("price", 0)),
                "volume": _to_float(row.get("volume", 0)),
                "change": _to_float(change_raw),
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
            "Average Volume": "Over 100K",
            "Performance": perf_filter,
        }
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()

        if df is None or df.empty:
            logger.warning("Finviz found no high gainers")
            return []

        _normalize_columns(df)

        tickers = df["ticker"].tolist() if "ticker" in df.columns else []
        logger.info(f"Finviz found {len(tickers)} stocks with >{min_gain_pct}% gain over {months} months")
        return tickers

    except Exception as e:
        logger.error(f"Finviz high gainer scan failed: {e}")
        return []
