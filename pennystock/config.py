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
STOCHRSI_PERIOD = 14  # StochRSI lookback on RSI values

# ── Screening ───────────────────────────────────────────────────────
STAGE1_KEEP_TOP_N = 50        # Stocks to pass from Stage 1 -> Stage 2
STAGE2_RETURN_TOP_N = 5       # Final picks returned to user
HISTORY_PERIOD = "6mo"        # Price history for technical analysis
SHORT_HISTORY_PERIOD = "3mo"  # Shorter window for recent patterns

# ═══════════════════════════════════════════════════════════════════
# LAYER 1: HARD KILL FILTERS
# Any single trigger = instant disqualification (score 0, excluded).
# These catch fundamentally broken companies that no amount of
# technical setup can save.
# ═══════════════════════════════════════════════════════════════════

# -- Kill: Going concern / auditor doubt (SEC EDGAR full-text) ------
KILL_GOING_CONCERN = True  # Search 10-K/10-Q for "going concern"
# Would have killed: ZONE (CleanCore Solutions)

# -- Kill: Delisting notice (news headline keywords) ---------------
KILL_DELISTING_KEYWORDS = [
    "delisting", "delist", "compliance notice", "noncompliance",
    "non-compliance", "listing requirements", "listing standards",
    "notice of deficiency", "reverse stock split", "proposed reverse split",
    "share consolidation",
]
# Would have killed: OPAD (Offerpad), DXST (reverse split vote)

# -- Kill: Fraud / SEC investigation (news headline keywords) ------
KILL_FRAUD_KEYWORDS = [
    "fraud", "sec investigation", "securities class action",
    "class action lawsuit", "restatement", "accounting irregularity",
    "securities fraud", "investor lawsuit", "shareholder lawsuit",
    "false claims act", "kickback", "department of justice", "doj lawsuit",
    "doj intervene", "sec halt", "trading halt", "trading suspended",
]
# Would have killed: AGL (Agilon), SLQT (DOJ kickbacks), SOPA (SEC halt)

# -- Kill: Core product failure (news headline patterns) -----------
KILL_FAILURE_KEYWORDS = [
    "failed trial", "phase 3 fail", "phase 2 fail", "phase 3 failure",
    "trial failure", "trial failed", "fda reject", "crl",
    "complete response letter", "discontinued", "terminated study",
    "clinical hold",
]
# Would have killed: QNCX (Quince Therapeutics)

# -- Kill: Shell company indicators (Yahoo Finance fundamentals) ---
KILL_SHELL_MAX_REVENUE = 100_000    # < $100K revenue = effectively zero
KILL_SHELL_MAX_MARKET_CAP = 5_000_000  # < $5M market cap
# Would have killed: QNCX (post-failure), AZI (Autozi)

# -- (DOWNGRADED to penalty) Extreme price decay from 52-week high -
# Now a scoring penalty, not a kill. See PENALTY_PRICE_DECAY below.

# -- Kill: Gross margin below threshold (non-pre-revenue) ---------
KILL_MIN_GROSS_MARGIN = 0.05        # 5% gross margin minimum
# Would have killed: AZI (1.6% gross margin)

# -- Kill: Cash runway under threshold ----------------------------
KILL_MIN_CASH_RUNWAY_YEARS = 0.5    # < 6 months cash = death spiral
# Computed as: total_cash / abs(operating_cashflow) when OCF < 0
# Would have killed: AGL (burning $20M/month with inadequate reserves)

# -- (DOWNGRADED to penalty) Max float (too large for penny setup) -
# Now a scoring penalty, not a kill. See PENALTY_EXCESSIVE_FLOAT below.

# -- Kill: Pre-revenue company with massive burn -------------------
KILL_PRE_REVENUE_MAX_REVENUE = 1_000_000    # < $1M revenue = pre-revenue
KILL_PRE_REVENUE_MIN_BURN = -50_000_000     # Burning > $50M/year
# Would have killed: GUTS ($3K revenue, -$86M/year burn)

# -- Kill: Already Pumped (catch BEFORE the pump, not after) -------
KILL_ALREADY_PUMPED_PCT = 100.0   # > 100% gain in recent days = already ran
KILL_ALREADY_PUMPED_DAYS = 5      # Lookback window (trading days)
# The single most important filter: we want stocks BEFORE the explosion

# ═══════════════════════════════════════════════════════════════════
# SCORING PENALTIES (downgraded from hard kills)
# These are "normal penny stock shadiness" -- bad signs that reduce
# the score but don't instantly disqualify. A stock can overcome
# these with strong setup + technicals + fundamentals.
# ═══════════════════════════════════════════════════════════════════

# Going concern -> score penalty instead of kill
PENALTY_GOING_CONCERN = 25          # Points deducted from final score (0-100)

# Delisting / compliance notice -> score penalty instead of kill
PENALTY_DELISTING_NOTICE = 20       # Points deducted from final score

# Extreme price decay (85%+ from 52w high) -> score penalty instead of kill
PENALTY_PRICE_DECAY = 15            # Points deducted from final score
PENALTY_PRICE_DECAY_THRESHOLD = 0.15  # Same threshold, but penalty not kill

# Recent reverse split -> score penalty instead of kill
PENALTY_REVERSE_SPLIT = 20          # Points deducted from final score

# Excessive float -> score penalty instead of kill
PENALTY_EXCESSIVE_FLOAT = 15        # Points deducted from final score
PENALTY_MAX_FLOAT = 100_000_000     # Same threshold as before

# ═══════════════════════════════════════════════════════════════════
# LAYER 2: POSITIVE SCORING WEIGHTS
# After passing kill filters, score stocks 0-100 across these
# weighted dimensions. Designed to rank RIME/ORKT-type setups highly.
# ═══════════════════════════════════════════════════════════════════

# -- Category-level weights (must sum to 1.0) ----------------------
WEIGHTS = {
    "setup":        0.40,   # Float, insider ownership, proximity-to-low, P/B
    "technical":    0.25,   # RSI, MACD, StochRSI, volume, price trend
    "fundamental":  0.25,   # Revenue growth, short interest, cash position
    "catalyst":     0.10,   # News-based catalysts
}

# -- Setup sub-weights (within the 40% setup allocation) -----------
SETUP_WEIGHTS = {
    "float_tightness":      0.35,   # Ultra-low float = explosive moves
    "insider_ownership":    0.25,   # High insider lock = less supply
    "proximity_to_low":     0.25,   # Near 52w low = max upside, min downside
    "price_to_book":        0.15,   # Below book value = margin of safety
}

# -- Technical sub-weights (within the 25% technical allocation) ---
TECH_WEIGHTS = {
    "rsi":          0.20,
    "macd":         0.20,
    "stochrsi":     0.20,   # NEW: StochRSI oversold detection
    "volume_spike": 0.15,
    "obv_trend":    0.10,
    "bollinger":    0.05,
    "price_trend":  0.10,
}

# -- Fundamental sub-weights (within the 25% fundamental alloc.) ---
FUNDAMENTAL_WEIGHTS = {
    "revenue_growth":   0.40,   # Strong growth = real business momentum
    "short_interest":   0.30,   # High SI + low float = squeeze fuel
    "cash_position":    0.30,   # Healthy cash = survival + opportunity
}

# ═══════════════════════════════════════════════════════════════════
# SCORING THRESHOLDS
# Specific breakpoints for each scoring dimension.
# ═══════════════════════════════════════════════════════════════════

# Float tightness scoring (shares)
FLOAT_THRESHOLDS = [
    (5_000_000,   100),   # < 5M shares: perfect squeeze setup
    (10_000_000,   85),   # 5-10M: excellent
    (20_000_000,   65),   # 10-20M: good
    (50_000_000,   40),   # 20-50M: average
    (float("inf"), 15),   # > 50M: diluted, hard to move
]

# Insider ownership scoring (fraction 0.0 - 1.0)
INSIDER_THRESHOLDS = [
    (0.50, 100),   # > 50%: insiders have massive skin in the game
    (0.30,  80),   # 30-50%: strong alignment
    (0.15,  60),   # 15-30%: decent
    (0.05,  40),   # 5-15%: moderate
    (0.00,  20),   # < 5%: insiders don't care
]

# Proximity to 52-week low scoring
# position = (price - 52w_low) / (52w_high - 52w_low), where 0 = at low, 1 = at high
PROXIMITY_LOW_THRESHOLDS = [
    (0.10,  95),   # Within 10% of bottom: maximum upside
    (0.25,  80),   # Near bottom quarter
    (0.40,  60),   # Lower half
    (0.60,  40),   # Middle
    (1.00,  15),   # Near 52w high: limited upside, chasing risk
]

# Price-to-book scoring
PB_THRESHOLDS = [
    (0.50, 100),   # Below half of book value: deep value
    (1.00,  75),   # Below book: value territory
    (2.00,  50),   # Reasonable
    (5.00,  30),   # Expensive for a penny stock
    (float("inf"), 15),
]

# Revenue growth scoring (YoY fraction, e.g. 1.0 = 100% growth)
REVENUE_GROWTH_THRESHOLDS = [
    (1.00, 100),   # > 100% YoY: explosive growth (RIME had 300%)
    (0.50,  80),   # 50-100%: strong
    (0.20,  65),   # 20-50%: healthy
    (0.00,  45),   # Flat
    (-0.20, 25),   # Declining
    (float("-inf"), 10),  # Collapsing (OPAD)
]

# Short interest as % of float scoring
SHORT_INTEREST_THRESHOLDS = [
    (0.20, 90),    # > 20%: heavy squeeze potential
    (0.10, 75),    # 10-20%: meaningful short pressure
    (0.05, 55),    # 5-10%: some shorts
    (0.00, 40),    # < 5%: minimal
]

# RSI scoring for penny stocks (optimal is 30-50 oversold bounce zone)
RSI_SCORE_MAP = {
    "30_50":  90,   # Oversold bounce zone (RIME was 42-47)
    "50_60":  70,   # Neutral with room to run
    "below_30": 55, # Deeply oversold, risky but could bounce
    "60_70":  40,   # Getting extended
    "above_70": 10, # Overbought, avoid
}

# StochRSI scoring (lower = more oversold = better entry)
STOCHRSI_THRESHOLDS = [
    (10,   95),    # Deeply oversold (RIME was 5.76)
    (20,   80),    # Oversold
    (40,   60),    # Neutral-low
    (60,   40),    # Neutral-high
    (80,   20),    # Overbought zone
    (100,   5),    # Extremely overbought
]

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
# These are scored on a spectrum, NOT kill filters.
# Kill-worthy keywords (fraud, delisting, etc.) are in the KILL_ lists above.
POSITIVE_CATALYSTS = [
    "contract", "partnership", "approval", "fda", "patent",
    "acquisition", "merger", "revenue", "earnings beat", "upgrade",
    "launch", "expansion", "grant", "license", "breakthrough",
    "deal", "award", "milestone", "collaboration", "agreement",
    "buyback", "repurchase", "analyst upgrade", "price target",
    "forbes", "featured in", "accelerating",
]
NEGATIVE_CATALYSTS = [
    "dilution", "offering", "lawsuit", "delisting", "bankruptcy",
    "sec investigation", "fraud", "default", "downgrade", "recall",
    "layoff", "suspend", "warning", "going concern",
    "resignation", "ceo depart", "ceo resign", "restatement",
    "shell company", "reverse split", "pump and dump",
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
