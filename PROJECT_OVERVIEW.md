# Penny Stock Analyzer - Project Overview

## Project Goal

Build an intelligent penny stock analyzer that identifies micro-cap stocks (priced $0.05-$1.00) with strong potential for short-term price increases using a combination of technical analysis, sentiment analysis, and fundamental indicators.

## What We've Accomplished

### 1. Core Architecture

**Technology Stack:**
- **Language:** Python 3.x
- **GUI Framework:** PyQt6 (dark theme, modern interface)
- **Data Sources:**
  - Finviz (primary stock screener)
  - Yahoo Finance (price/volume data)
  - Reddit (sentiment analysis - 12 subreddits)
  - StockTwits (social sentiment)
- **Database:** SQLite for persistent storage
- **Logging:** Loguru for comprehensive logging

**Project Structure:**
```
PENNYSTOCK/
├── launch.bat                      # Main launcher (auto-installs dependencies)
├── pennystock_picker/
│   ├── unified_penny_analyzer.py   # Main GUI application (500 lines)
│   ├── enhanced_stock_picker.py    # Two-stage filtering algorithm
│   ├── sentiment_engine.py         # Reddit/StockTwits sentiment analysis
│   ├── finviz_screener.py          # Stock discovery via Finviz
│   ├── quick_analysis.py           # Pattern learning from winners
│   ├── comprehensive_penny_analysis.py  # Deep stock analysis
│   ├── config.py                   # Configuration management
│   ├── database.py                 # SQLite database interface
│   └── requirements.txt            # Auto-install dependencies
```

### 2. Two-Stage Filtering Algorithm (Core Innovation)

**Problem Solved:**
- Analyzing 400+ stocks with deep analysis takes 2+ hours
- Most stocks don't pass basic technical filters
- Wasting time on sentiment analysis for non-starters

**Solution: Smart Two-Stage Filter**

#### Stage 1: Technical Filter (Fast - ~14 minutes)
**Input:** All stocks from Finviz screener (typically 400-500 stocks)

**Process:**
1. Download 3-month price history for each stock
2. Calculate technical indicators:
   - **RSI (Relative Strength Index):** Target ~50 (balanced, not overbought)
   - **Volume Spike:** Current volume vs 50-day average (target: 7.7x)
   - **Price Trend:** 20-day price change percentage (target: +67.6%)
3. Score each stock on technical merit (0-100)
4. Sort by technical score

**Output:** Top 50 stocks with strongest technical signals

**Technical Scoring Formula:**
```python
technical_score = (
    volume_score * 0.5 +    # 50% weight - CRITICAL for penny stocks
    rsi_score * 0.3 +        # 30% weight
    trend_score * 0.2        # 20% weight
)

# Scoring functions:
volume_score = max(0, 100 - abs(volume_spike - 7.7) * 10)
rsi_score = max(0, 100 - abs(current_rsi - 50) * 2)
trend_score = max(0, 100 - abs(price_trend - 67.6) / 2)
```

**Why These Targets?**
- Learned from analyzing historical winners (OPTT case study)
- Volume spike of 7.7x average = institutional/retail interest
- RSI ~50 = not overbought, room to run
- Price trend +67.6% = existing momentum

#### Stage 2: Deep Analysis (Comprehensive - ~15 minutes)
**Input:** Top 50 stocks from Stage 1

**Process:** Apply ALL analysis factors:

1. **Social Sentiment (25% weight)**
   - Reddit sentiment across 12 subreddits:
     - r/pennystocks, r/RobinHoodPennyStocks, r/Shortsqueeze
     - r/StockMarket, r/stocks, r/wallstreetbets
     - r/smallstreetbets, r/Daytrading, r/StocksAndTrading
     - r/stockpicks, r/UndervaluedStonks, r/Canadapennystocks
   - StockTwits sentiment and message volume
   - Mention count scoring (more mentions = more attention)

2. **Revenue Momentum (20% weight)**
   - Quarterly revenue growth rate
   - Growth trend direction (accelerating/decelerating)
   - Target: Companies like OPTT with rapid revenue growth

3. **Recent Catalysts (15% weight)**
   - Scan last 10 news articles for keywords:
     - Contracts, partnerships, approvals, launches
     - FDA approvals, patent grants, acquisitions
   - Positive sentiment in news headlines

4. **Financial Health (10% weight)**
   - Cash position (higher = better)
   - Debt-to-equity ratio (lower = better)
   - Ensures company can survive and execute

**Combined Scoring Formula:**
```python
final_score = (
    technical_score * 0.30 +      # From Stage 1
    sentiment_score * 0.25 +      # Reddit + StockTwits
    revenue_score * 0.20 +        # Revenue growth
    catalyst_score * 0.15 +       # News catalysts
    financial_score * 0.10        # Financial health
)
```

**Output:** Top 1-5 stocks ranked by final score

### 3. GUI Features

**Simplified 2-Tab Interface:**

#### Tab 1: "Analyze Current Winners"
- **Purpose:** Learn from stocks that ARE pumping right now
- **Process:**
  1. Finds stocks with 6-month gains >100%
  2. Analyzes their common patterns
  3. Updates algorithm targets (RSI, volume spike, price trend)
- **Usage:** Run weekly to keep algorithm current
- **Time:** 5-10 minutes

#### Tab 2: "Pick Top Stocks"
- **Purpose:** Find 1-5 stocks to buy today
- **Process:** Runs two-stage filtering algorithm
- **Adjustable Settings:**
  - Min/Max Price (default: $0.05-$1.00)
  - Min Volume (default: 50,000)
  - Top N stocks to return (default: 5)
- **Usage:** Run daily for fresh picks
- **Time:** ~30 minutes

**UI Features:**
- Dark theme for eye comfort
- Real-time Debug Console showing progress
- Sortable results table (click any column to sort)
- Progress indicators with ETA
- Color-coded log levels (INFO, WARNING, ERROR, SUCCESS)

### 4. Auto-Install System (Three Layers)

**Problem:** Users had to manually install dependencies via command line

**Solution:**

**Layer 1: requirements.txt**
- Standard Python dependency file
- Includes finvizfinance, yfinance, requests, PyQt6, etc.
- Installed automatically by launch.bat

**Layer 2: launch.bat checks**
- Detects missing packages before launch
- Auto-installs if missing
- Shows progress in console

**Layer 3: GUI auto-installer**
- Fallback if user runs Python directly
- Uses subprocess to pip install
- Shows progress in Debug Console

**Result:** Zero manual setup required!

### 5. Stock Discovery System (Multi-Tier Fallback)

**Tier 1: Finviz Screener (Primary - Fast & Reliable)**
```python
from finvizfinance.screener.overview import Overview

filters = {
    'Price': 'Under $1',
    'Average Volume': 'Over 50K'
}
screener = Overview()
screener.set_filter(filters_dict=filters)
stocks = screener.screener_view()
# Returns 400-500 real, actively trading stocks
```

**Tier 2: Yahoo Finance (Backup)**
- Batch download in groups of 50 to avoid rate limits
- Slower but reliable fallback

**Tier 3: Curated List (Guaranteed)**
- Hardcoded list of 200+ known penny stocks
- Never fails, always returns stocks
- Used when both Finviz and Yahoo fail

### 6. Pattern Learning System

**File:** `quick_analysis.py`

**Purpose:** Learn what winners look like RIGHT NOW

**Process:**
1. Screen for stocks that rose >100% in last 6 months
2. Download price/volume history for each winner
3. Calculate technical indicators at time of rise:
   - What was their RSI?
   - What was their volume spike?
   - What was their price trend?
4. Average these values → winning pattern
5. Save to `winning_pattern.json`

**Pattern File Format:**
```json
{
  "average_conditions": {
    "rsi": 50.0,
    "volume_spike_max": 7.7,
    "price_trend_pct": 67.6
  },
  "analysis_date": "2026-02-07",
  "winners_analyzed": 15
}
```

**Usage in Picker:**
- Enhanced picker loads this pattern
- Scores stocks based on similarity to winners
- Pattern updates weekly to stay current

### 7. Sentiment Analysis Engine

**File:** `sentiment_engine.py`

**Reddit Analysis:**
- Uses `requests` to scrape (no API key needed)
- Searches 12 subreddits for ticker mentions
- Analyzes sentiment: positive/negative/neutral
- Counts total mentions across all subreddits
- 1-second delay between requests to avoid rate limiting

**StockTwits Analysis:**
- Uses public StockTwits API
- Gets recent messages about ticker
- Sentiment score from message sentiment
- Message volume indicates attention

**Combined Sentiment Score:**
```python
sentiment_score = (
    (reddit_sentiment + stocktwits_sentiment) / 2 * 0.7 +
    min(100, mention_count * 2) * 0.3
)
```

### 8. Performance Optimizations

**Problem:** Original implementation took 2+ hours for 422 stocks

**Optimizations Applied:**

1. **Two-Stage Filtering**
   - Don't waste time on low-quality stocks
   - Deep analysis only on promising candidates
   - Result: 2 hours → 30 minutes (4x speedup)

2. **Batch Downloading**
   - Yahoo Finance requests in groups of 50
   - Prevents rate limiting
   - Reduces timeout errors

3. **Progress Logging**
   - Updates every 10 stocks in Stage 1
   - Updates every 5 stocks in Stage 2
   - Shows ETA and processing rate
   - User sees system is working, not frozen

4. **Graceful Degradation**
   - Finviz fails → try Yahoo Finance
   - Yahoo fails → use curated list
   - Sentiment fails → use neutral score (50)
   - System never crashes, always returns results

5. **Random Delays**
   - 0.1-0.2 second delay between stocks
   - Avoids triggering rate limits
   - Appears more "human-like" to APIs

### 9. Data Sources & APIs

**Finviz (Stock Screening):**
- Library: `finvizfinance>=1.3.0`
- Free tier, no API key
- Fast and reliable
- Best for initial screening

**Yahoo Finance (Price Data):**
- Library: `yfinance>=0.2.28`
- Free, no API key
- Historical price/volume data
- Quarterly financials, balance sheets
- Company news

**Reddit (Sentiment):**
- Method: Direct HTTP requests to reddit.com
- No API key needed (uses public JSON endpoints)
- Format: `https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&limit=100`
- Rate limit: ~30 requests/minute per IP

**StockTwits (Social Sentiment):**
- Free public API
- Format: `https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json`
- Rate limit: Not officially documented, but generous

### 10. Key Files & Their Purposes

**Core Application:**
- `unified_penny_analyzer.py` (500 lines)
  - PyQt6 GUI application
  - 2-tab interface (Analyze, Pick)
  - WorkerThread for async operations
  - Debug Console for real-time logging
  - Auto-dependency checking

**Stock Pickers:**
- `enhanced_stock_picker.py` (600+ lines)
  - Two-stage filtering algorithm
  - Technical analysis calculations
  - Score combination logic
  - Pattern loading/application

**Analysis Engines:**
- `sentiment_engine.py` (420 lines)
  - Reddit scraping (12 subreddits)
  - StockTwits API integration
  - Sentiment scoring algorithms

- `comprehensive_penny_analysis.py` (586 lines)
  - Deep stock analysis
  - Revenue momentum calculation
  - Financial health scoring
  - News catalyst detection

- `quick_analysis.py` (246 lines)
  - Winner pattern learning
  - 6-month performance screening
  - Pattern extraction and saving

**Data Sources:**
- `finviz_screener.py` (256 lines)
  - Finviz API integration
  - Price/volume filtering
  - Exact price range enforcement

- `active_penny_stocks.py` (161 lines)
  - Curated ticker list fallback
  - Yahoo Finance batch download
  - Guaranteed stock source

**Support:**
- `config.py` (97 lines)
  - Centralized configuration
  - API endpoints
  - Timeout settings

- `database.py` (435 lines)
  - SQLite interface
  - Analysis history storage
  - Results persistence

## Technical Indicators Explained

### 1. RSI (Relative Strength Index)
**What it is:** Momentum oscillator measuring speed and magnitude of price changes

**Formula:**
```python
delta = price.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

**Interpretation:**
- 0-30: Oversold (potential buy)
- 30-70: Neutral
- 70-100: Overbought (potential sell)
- **Our target: ~50** (balanced, not overbought, room to run)

### 2. Volume Spike
**What it is:** Current volume compared to average volume

**Formula:**
```python
volume_spike = current_volume / average_volume_50_days
```

**Interpretation:**
- 1.0x = Normal volume
- 2-3x = Increased interest
- 5-10x = Significant interest (institutional/retail buying)
- **Our target: 7.7x** (strong buying pressure from winners analysis)

**Why it matters:**
- Penny stocks need volume to move
- High volume = liquidity = easier to buy/sell
- Unusual volume = something happening (news, catalyst, pump)

### 3. Price Trend
**What it is:** Price change over recent period (20 days)

**Formula:**
```python
price_trend = ((current_price - price_20_days_ago) / price_20_days_ago) * 100
```

**Interpretation:**
- Negative = Downtrend
- 0-20% = Slight uptrend
- 20-50% = Moderate uptrend
- 50%+ = Strong uptrend
- **Our target: +67.6%** (strong existing momentum from winners)

**Why it matters:**
- Momentum tends to continue
- Stocks in uptrend more likely to keep rising
- We want to catch the wave, not the falling knife

## Algorithm Flow (Step-by-Step)

### Daily Use Flow

**Step 1: User Launches Application**
```bash
# User double-clicks launch.bat
launch.bat
  → Creates Python venv if needed
  → Installs dependencies from requirements.txt
  → Checks for finvizfinance specifically
  → Launches unified_penny_analyzer.py
```

**Step 2: User Clicks "Pick Top Stocks"**
```
GUI (unified_penny_analyzer.py)
  → Gets settings from UI (min_price, max_price, min_volume, top_n)
  → Creates WorkerThread
  → Calls enhanced_stock_picker.enhanced_pick_stocks()
  → Shows "Picking..." in Debug Console
```

**Step 3: Stage 1 - Technical Filter**
```
enhanced_stock_picker.py::enhanced_pick_stocks()
  → Load winning pattern from winning_pattern.json

  → Get stocks from Finviz:
      finviz_screener.get_finviz_penny_stocks(0.05, 1.00, 50000)
        → Finviz API query: Price < $1, Volume > 50K
        → Returns ~422 real, active stocks

  → For each of 422 stocks:
      → Download 3-month price history (yfinance)
      → Calculate RSI (14-period)
      → Calculate volume spike (current vs 50-day avg)
      → Calculate price trend (20-day change)
      → Score technical pattern (0-100)
      → Save to technical_candidates list
      → Log progress every 10 stocks with ETA

  → Sort technical_candidates by technical_score
  → Take top 50 stocks
  → Log Stage 1 results
```

**Step 4: Stage 2 - Deep Analysis (Top 50 Only)**
```
For each of top 50 stocks:
  → Sentiment Analysis:
      sentiment_engine.get_sentiment(symbol)
        → Reddit scraping (12 subreddits)
        → StockTwits API call
        → Combine into sentiment_score (0-100)

  → Revenue Momentum:
      get_revenue_momentum_score(symbol)
        → Yahoo Finance quarterly_financials
        → Calculate growth rate
        → Score based on growth trend (0-100)

  → News Catalysts:
      get_recent_news_score(symbol)
        → Yahoo Finance news (last 10 articles)
        → Scan for catalyst keywords
        → Score based on positive news (0-100)

  → Financial Health:
      get_financial_health_score(symbol)
        → Yahoo Finance balance sheet
        → Calculate cash position
        → Calculate debt ratio
        → Score financial stability (0-100)

  → Combine all scores:
      final_score = (
          technical_score * 0.30 +
          sentiment_score * 0.25 +
          revenue_score * 0.20 +
          catalyst_score * 0.15 +
          financial_score * 0.10
      )

  → Save to all_scored_stocks list
  → Log progress every 5 stocks
```

**Step 5: Return Results**
```
  → Sort all_scored_stocks by final_score (highest first)
  → Take top N (default: 5)
  → Return to GUI

GUI receives results:
  → Display in table (symbol, price, volume, scores)
  → Show in Debug Console
  → User can sort by any column
  → User can research stocks and decide to trade
```

### Weekly Use Flow (Pattern Learning)

**User Clicks "Analyze Current Winners"**
```
quick_analysis.py::run_analysis()
  → Screen Finviz for stocks with 6-month gain >100%
  → Download history for each winner
  → Calculate their technical indicators when they rose:
      → What was their RSI?
      → What was their volume spike?
      → What was their price trend?
  → Average all winners' values
  → Save to winning_pattern.json
  → Next daily run uses updated pattern
```

## Key Insights & Lessons Learned

### 1. Volume is King for Penny Stocks
- 50% weight in technical score
- Penny stocks need volume to move price
- Low volume = wide spreads = hard to exit
- Volume spike = attention = potential for gains

### 2. Two-Stage Filtering is Essential
- Don't waste time on stocks that fail basic tests
- 90% of stocks eliminated in Stage 1 (fast)
- Deep analysis only on 10% (50 stocks)
- Result: 4x faster without losing quality

### 3. Pattern Learning Keeps Algorithm Current
- Markets change, patterns shift
- What worked 6 months ago may not work today
- Weekly pattern updates keep algorithm adaptive
- Learn from current winners, not historical data

### 4. Graceful Degradation is Critical
- APIs fail, rate limits hit, network issues occur
- Always have fallback options
- Never crash, always return something
- User experience > perfect data

### 5. Transparency Builds Trust
- Show user what's happening (Debug Console)
- Log every step, every API call, every decision
- User can verify results manually
- Clear progress indicators (not a black box)

### 6. Sentiment is Noisy but Valuable
- Reddit/StockTwits full of hype and pumps
- BUT: Attention often precedes price movement
- Combined with technicals = powerful signal
- Don't rely on sentiment alone

### 7. Finviz > Yahoo Finance for Discovery
- Finviz finds REAL, actively trading stocks
- Yahoo Finance ticker lists are stale
- Many Yahoo tickers are delisted, frozen, or OTC
- Finviz = higher quality input = better output

## Installation & Setup

### Requirements
- Python 3.8+
- Windows (launch.bat) or Linux/Mac (modify launcher)
- Internet connection
- ~500MB disk space

### Dependencies (Auto-Installed)
```
finvizfinance>=1.3.0      # Stock screening
yfinance>=0.2.28          # Price/financial data
requests>=2.31.0          # HTTP requests
beautifulsoup4>=4.12.0    # HTML parsing
PyQt6>=6.6.0              # GUI framework
loguru>=0.7.2             # Logging
pandas>=2.1.0             # Data manipulation
numpy>=1.26.0             # Numerical computing
```

### First Run
```bash
# 1. Download project
git clone <your-repo-url>
cd PENNYSTOCK

# 2. Launch (auto-installs everything)
launch.bat   # Windows
# or
./launch.sh  # Linux/Mac

# 3. Wait for GUI to open (first launch slower - installing deps)

# 4. Optional: Run "Analyze Current Winners" to learn patterns (10 min)

# 5. Run "Pick Top Stocks" to get daily picks (30 min)
```

## Expected Results

### Example Output

**Stage 1 Complete:**
```
✅ Stage 1 complete: 422 stocks analyzed
📊 Top technical scores: ['OPTT:95.3', 'ABVC:89.7', 'MEGL:87.2', 'PXMD:85.1', 'ADTX:82.9']
```

**Stage 2 Complete:**
```
🔬 STAGE 2: Deep analysis on top 50 technical candidates...
  Stage 2 progress: 5/50...
  Stage 2 progress: 10/50...
  ...
✅ Stage 2 complete in 847s

================================================================================
RESULTS: Top 5 Stocks
================================================================================

#1. OPTT - $0.73 (Final Score: 87.5)
    Technical: 95.3 | Sentiment: 82.1 | Revenue: 89.0 | Catalyst: 75.0 | Financial: 68.0

    Metrics:
    - Volume Spike: 8.2x average
    - RSI: 48.5 (balanced)
    - Price Trend: +72.3% (20-day)
    - Catalysts: 3 recent news items
    - Reddit Mentions: 47 across subreddits
    - Revenue Growth: +215% QoQ

    Reasoning: Strong technical signals (high volume, rising price), positive social
    sentiment, explosive revenue growth, recent contract news. All factors aligned.

#2. ABVC - $0.58 (Final Score: 84.2)
    Technical: 89.7 | Sentiment: 78.5 | Revenue: 81.0 | Catalyst: 80.0 | Financial: 72.0
    ...

#3. MEGL - $0.45 (Final Score: 81.8)
    ...

#4. PXMD - $0.82 (Final Score: 79.3)
    ...

#5. ADTX - $0.91 (Final Score: 77.6)
    ...
```

### What Makes a Good Pick?

**Strong Pick Characteristics:**
- ✅ Technical Score >85 (volume spike >5x, RSI 40-60, price rising)
- ✅ Sentiment Score >70 (Reddit/StockTwits buzz)
- ✅ Revenue Score >75 (growing revenue)
- ✅ Catalyst Score >60 (recent positive news)
- ✅ Financial Score >50 (not bankrupt)
- ✅ Final Score >75

**Red Flags:**
- ❌ Technical Score <50 (no volume, poor price action)
- ❌ Sentiment Score <30 (no one talking about it)
- ❌ Revenue Score <30 (declining revenue)
- ❌ Financial Score <30 (massive debt, no cash)
- ❌ Final Score <50

## Limitations & Disclaimers

### ⚠️ CRITICAL WARNINGS

**This is NOT financial advice!**
- Tool is for research and education only
- Always do your own due diligence
- Consult licensed financial advisor before trading
- Past performance ≠ future results

**Penny stocks are EXTREMELY risky:**
- Can lose 100% of investment
- Extreme volatility (±50% in a day)
- Low liquidity (hard to sell)
- Pump-and-dump schemes common
- Many companies are unprofitable
- Delisting risk

**Algorithm limitations:**
- Based on historical patterns (may not predict future)
- Sentiment can be manipulated (bots, shills)
- News catalysts may be overhyped
- Technical indicators lag (past data)
- No guarantee of accuracy

### Known Issues

1. **Rate Limiting:**
   - Reddit: ~30 requests/minute
   - Yahoo Finance: Soft limits, varies
   - Solution: Random delays, batch requests

2. **Data Quality:**
   - Yahoo Finance data sometimes stale
   - Reddit sentiment can be manipulated
   - News headlines can be misleading
   - Solution: Multi-source verification

3. **Performance:**
   - Stage 2 still takes ~15 minutes (50 stocks)
   - Reddit scraping is slowest part
   - Solution: Could parallelize, add caching

4. **Coverage:**
   - Only analyzes stocks Finviz finds
   - Misses some OTC/pink sheet stocks
   - Solution: Manual ticker additions

## Future Enhancements

### Planned Features

1. **Result Caching (1 hour)**
   - Save Stage 1 results for 60 minutes
   - Rerun Stage 2 only if needed
   - Speed up repeated queries

2. **Parallel Processing**
   - Analyze 5-10 stocks simultaneously in Stage 2
   - Use multiprocessing or threading
   - Reduce Stage 2 time from 15 min → 3-5 min

3. **Backtesting**
   - Test algorithm on historical data
   - Measure win rate, average gain, drawdown
   - Optimize scoring weights

4. **Portfolio Tracking**
   - Save picks to watchlist
   - Track performance over time
   - Show win/loss statistics

5. **Alert System**
   - Email/SMS when high-scoring stock found
   - Price alerts for watched stocks
   - News alerts for holdings

6. **Advanced Filters**
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Stochastic oscillator
   - Fibonacci retracements

7. **Machine Learning**
   - Train model on historical winners
   - Predict probability of gain
   - Feature importance analysis

8. **API Mode**
   - REST API for programmatic access
   - Integrate with trading bots
   - Scheduled daily runs

## Case Study: OPTT (Ocean Power Technologies)

**Why OPTT was studied:**
- Massive gain: $0.20 → $1.20 in 6 months (+500%)
- Used as reference for pattern learning
- Informed algorithm targets

**OPTT's Winning Pattern:**
- Volume spike: 7.7x average (CRITICAL)
- RSI: ~48 when it started rising
- Price trend: Already up 67% when picked
- Catalyst: Navy contract announcement
- Revenue: Growing rapidly QoQ
- Sentiment: Reddit/StockTwits buzz

**What we learned:**
- Volume precedes price movement
- RSI ~50 is ideal entry (not overbought)
- Existing momentum tends to continue
- Catalysts (news) fuel the fire
- Social sentiment amplifies moves

**How it influenced algorithm:**
- Set volume spike target to 7.7x
- Set RSI target to 50
- Set price trend target to +67.6%
- Weighted volume at 50% in technical score
- Added catalyst detection (news scanning)

## How to Recreate This Project

### Phase 1: Basic Structure (Day 1)

1. **Setup:**
   ```bash
   mkdir pennystock_analyzer
   cd pennystock_analyzer
   git init
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Create requirements.txt:**
   ```
   finvizfinance>=1.3.0
   yfinance>=0.2.28
   requests>=2.31.0
   beautifulsoup4>=4.12.0
   PyQt6>=6.6.0
   loguru>=0.7.2
   pandas>=2.1.0
   numpy>=1.26.0
   ```

3. **Install:**
   ```bash
   pip install -r requirements.txt
   ```

### Phase 2: Stock Discovery (Day 2)

1. **Create finviz_screener.py:**
   - Import finvizfinance
   - Filter for price < $1, volume > 50K
   - Return list of tickers

2. **Test:**
   ```python
   from finviz_screener import get_finviz_penny_stocks
   stocks = get_finviz_penny_stocks(0.05, 1.0, 50000)
   print(f"Found {len(stocks)} stocks")
   ```

### Phase 3: Technical Analysis (Day 3-4)

1. **Create technical_analyzer.py:**
   - Use yfinance to download history
   - Calculate RSI (14-period)
   - Calculate volume spike
   - Calculate price trend
   - Return scores

2. **Test on single stock:**
   ```python
   import yfinance as yf
   ticker = yf.Ticker("OPTT")
   hist = ticker.history(period="3mo")
   # Calculate RSI, volume spike, price trend
   ```

### Phase 4: Sentiment Analysis (Day 5-6)

1. **Create sentiment_engine.py:**
   - Reddit scraping via requests
   - StockTwits API integration
   - Sentiment scoring logic

2. **Test:**
   ```python
   from sentiment_engine import SentimentEngine
   engine = SentimentEngine()
   data = engine.get_sentiment("OPTT")
   print(f"Sentiment: {data['combined_score']}")
   ```

### Phase 5: Enhanced Picker (Day 7-8)

1. **Create enhanced_stock_picker.py:**
   - Implement two-stage filtering
   - Combine all scoring factors
   - Sort and return top N

2. **Test:**
   ```python
   from enhanced_stock_picker import EnhancedStockPicker
   picker = EnhancedStockPicker()
   results = picker.enhanced_pick_stocks(top_n=5)
   for stock in results:
       print(f"{stock['symbol']}: {stock['final_score']}")
   ```

### Phase 6: GUI (Day 9-10)

1. **Create unified_penny_analyzer.py:**
   - PyQt6 main window
   - 2-tab layout (Analyze, Pick)
   - Debug Console widget
   - WorkerThread for async operations

2. **Test:**
   ```bash
   python unified_penny_analyzer.py
   ```

### Phase 7: Pattern Learning (Day 11)

1. **Create quick_analysis.py:**
   - Screen for 6-month winners
   - Calculate average technical indicators
   - Save to winning_pattern.json

2. **Integrate with picker:**
   - Load pattern in enhanced_stock_picker
   - Use pattern as scoring targets

### Phase 8: Polish & Deploy (Day 12-14)

1. **Create launch.bat/launch.sh:**
   - Auto-install dependencies
   - Launch GUI

2. **Documentation:**
   - README.md
   - QUICK_START.md
   - API documentation

3. **Testing:**
   - Test all features end-to-end
   - Test error handling
   - Test with various stock symbols

### Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Finds 400+ penny stocks from Finviz
- ✅ Calculates technical scores
- ✅ Returns top 5 picks in <30 minutes
- ✅ GUI shows results in table
- ✅ Auto-installs dependencies

**Full Feature Set:**
- ✅ Two-stage filtering algorithm
- ✅ Sentiment analysis (Reddit + StockTwits)
- ✅ Revenue momentum scoring
- ✅ News catalyst detection
- ✅ Financial health scoring
- ✅ Pattern learning from winners
- ✅ GUI with Debug Console
- ✅ Progress indicators with ETA

## Resources & References

### Documentation
- **Finviz API:** https://finvizfinance.readthedocs.io/
- **yfinance:** https://pypi.org/project/yfinance/
- **PyQt6:** https://doc.qt.io/qtforpython-6/
- **Pandas:** https://pandas.pydata.org/docs/
- **NumPy:** https://numpy.org/doc/

### Learning Resources
- **RSI Explained:** https://www.investopedia.com/terms/r/rsi.asp
- **Volume Analysis:** https://www.investopedia.com/articles/technical/02/010702.asp
- **Penny Stock Risks:** https://www.sec.gov/investor/alerts/ia_pennystocks.html

### Community
- r/pennystocks (Reddit)
- r/StockMarket (Reddit)
- StockTwits.com

## Support & Contributing

### Getting Help
1. Check documentation files (START_HERE_SIMPLE.txt, etc.)
2. Review Debug Console output for errors
3. Check TROUBLESHOOTING.txt

### Contributing
1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

### Reporting Issues
- Include error messages
- Include Debug Console output
- Describe steps to reproduce
- Include Python version, OS version

## License & Credits

**License:** MIT (or your choice)

**Credits:**
- Finviz for stock screening data
- Yahoo Finance for financial data
- Reddit communities for insights
- PyQt6 for GUI framework

**Disclaimer:**
This software is provided "as is" without warranty of any kind.
Use at your own risk. Not financial advice. Always do your own research.

---

## Summary

This project successfully built an intelligent penny stock analyzer that:

1. **Discovers** 400+ real, active penny stocks via Finviz
2. **Filters** using two-stage algorithm (technical → deep analysis)
3. **Scores** stocks on 5 factors (technical, sentiment, revenue, catalysts, financials)
4. **Returns** top 1-5 picks in ~30 minutes
5. **Learns** from current winners to stay adaptive
6. **Auto-installs** all dependencies for zero-friction setup

The key innovation is the **two-stage filtering algorithm**, which achieves 4x speedup
without sacrificing analysis quality by applying expensive deep analysis only to stocks
that pass basic technical filters first.

**Total Development:** ~14 days for MVP, ~30 days for full feature set

**Lines of Code:** ~3,500 across 15 Python files

**Dependencies:** 8 core libraries, all auto-installed

**Platform:** Windows/Linux/Mac with Python 3.8+

Ready to find winning penny stocks! 🚀📈
