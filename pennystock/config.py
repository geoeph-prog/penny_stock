"""Central configuration for the Penny Stock Analyzer."""

# ── Price & Volume Filters ──────────────────────────────────────────
MIN_PRICE = 0.05
MAX_PRICE = 1.00
MIN_VOLUME = 50_000

# ── Technical Analysis Parameters ───────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
VOLUME_AVG_PERIOD = 50
STOCHASTIC_PERIOD = 14

# ── Screening ───────────────────────────────────────────────────────
STAGE1_KEEP_TOP_N = 50        # Stocks to pass from Stage 1 -> Stage 2
STAGE2_RETURN_TOP_N = 5       # Final picks returned to user
HISTORY_PERIOD = "6mo"        # Price history for technical analysis
SHORT_HISTORY_PERIOD = "3mo"  # Shorter window for recent patterns

# ── Scoring Weights (Stage 2 composite) ─────────────────────────────
# These are initial guesses -- backtesting will optimize them
WEIGHTS = {
    "technical":    0.25,
    "sentiment":    0.20,
    "fundamental":  0.15,
    "catalyst":     0.15,
    "market_ctx":   0.10,
    "short_squeeze": 0.15,
}

# ── Technical Sub-Weights ───────────────────────────────────────────
TECH_WEIGHTS = {
    "rsi":          0.15,
    "macd":         0.15,
    "volume_spike": 0.25,
    "obv_trend":    0.10,
    "bollinger":    0.10,
    "stochastic":   0.10,
    "price_trend":  0.15,
}

# ── Reddit Scraping ─────────────────────────────────────────────────
REDDIT_SUBREDDITS = [
    "pennystocks",
    "RobinHoodPennyStocks",
    "Shortsqueeze",
    "StockMarket",
    "stocks",
    "wallstreetbets",
    "smallstreetbets",
    "Daytrading",
    "StocksAndTrading",
    "stockpicks",
    "UndervaluedStonks",
]
REDDIT_POSTS_PER_SUB = 100
REDDIT_DELAY_SECONDS = 2.0  # Delay between requests

# ── StockTwits ──────────────────────────────────────────────────────
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

# ── Twitter (optional -- requires twikit + throwaway account) ───────
TWITTER_ENABLED = False
TWITTER_COOKIES_FILE = "twitter_cookies.json"
TWITTER_TWEETS_PER_TICKER = 30

# ── Catalyst Keywords ──────────────────────────────────────────────
POSITIVE_CATALYSTS = [
    "contract", "partnership", "approval", "fda", "patent",
    "acquisition", "merger", "revenue", "earnings beat", "upgrade",
    "launch", "expansion", "grant", "license", "breakthrough",
    "deal", "award", "milestone", "collaboration", "agreement",
]
NEGATIVE_CATALYSTS = [
    "dilution", "offering", "lawsuit", "delisting", "bankruptcy",
    "sec investigation", "fraud", "default", "downgrade", "recall",
    "loss", "layoff", "suspend", "warning", "debt",
]

# ── Sentiment NLP ───────────────────────────────────────────────────
# Financial term adjustments for VADER
FINANCIAL_LEXICON_UPDATES = {
    "moon": 2.5,
    "mooning": 2.5,
    "rocket": 2.0,
    "squeeze": 1.5,
    "bullish": 2.0,
    "bearish": -2.0,
    "pump": -1.0,   # Often negative context in penny stocks
    "dump": -2.5,
    "scam": -3.0,
    "dilution": -2.5,
    "bagholding": -1.5,
    "bagholder": -1.5,
    "calls": 1.0,
    "puts": -1.0,
    "long": 1.0,
    "short": -1.0,
    "buy": 1.5,
    "sell": -1.5,
    "dip": 0.5,     # "buy the dip" is positive
    "breakout": 2.0,
    "catalyst": 1.5,
    "undervalued": 1.5,
    "overvalued": -1.5,
    "pt": 1.0,      # Price target
    "dd": 1.0,      # Due diligence (usually positive context)
    "fda": 1.0,
    "revenue": 0.5,
    "profit": 1.5,
    "loss": -1.5,
    "debt": -1.0,
    "offering": -2.0,
}

# ── Market Context ──────────────────────────────────────────────────
VIX_TICKER = "^VIX"
FEAR_GREED_FAVORABLE_RANGE = (25, 60)  # Buy when fearful but not panic
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
}

# ── Backtesting ─────────────────────────────────────────────────────
BACKTEST_LOOKBACK_MONTHS = 12
BACKTEST_HOLD_DAYS = [5, 10, 20, 30]  # Evaluate returns at these horizons
BACKTEST_WINNER_THRESHOLD = 0.20      # 20% gain = winner
BACKTEST_LOSER_THRESHOLD = -0.15      # 15% loss = loser

# ── Caching ─────────────────────────────────────────────────────────
CACHE_DIR = ".cache"
CACHE_TTL_HOURS = 1  # Re-fetch after this many hours

# ── Rate Limiting ───────────────────────────────────────────────────
YAHOO_BATCH_SIZE = 20          # Download this many tickers at once
YAHOO_DELAY_SECONDS = 0.5     # Delay between batches
YAHOO_MAX_RETRIES = 3
