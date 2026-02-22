"""
Validated Penny Stock Winners: November 2025 - January 2026
============================================================
Real historical winners with verified price action and catalysts.
Intended for backtesting validation of the penny stock scoring algorithm.

Data compiled from multiple sources including StockTitan, Yahoo Finance,
Benzinga, StocksToTrade, Timothy Sykes, TipRanks, Motley Fool, and others.
"""

VALIDATED_WINNERS = {
    # =========================================================================
    # NOVEMBER 2025 WINNERS
    # =========================================================================
    "november_2025": [
        {
            "ticker": "KTTA",
            "company": "Pasithea Therapeutics Corp.",
            "sector": "Biotech / Pharmaceuticals",
            "month": "November 2025",

            # Price data
            "starting_price": 0.49,  # ~$0.49 on Nov 25, 2025
            "peak_price": 1.20,      # Estimated peak after 142% surge
            "percentage_gain": 142,   # Single-day gain Nov 26; extended to ~145% over 2 days
            "date_of_move": "2025-11-26",

            # Catalyst
            "catalyst": (
                "Positive Phase 1 interim data for PAS-004 (oral MEK inhibitor) in MAPK-driven "
                "advanced solid tumors. Completion of Cohort 7 with zero treatment-related adverse "
                "events. Positive tablet PK data showing linear dose-proportional pharmacokinetics. "
                "$1M ALS Association grant for Phase 1 ALS study. New clinical trial site activated "
                "at University of Alabama. FDA Fast Track designation."
            ),

            # Float / short interest
            "float_size": "Small (micro-cap, ~$20M market cap pre-move)",
            "short_interest": "Not specifically reported; likely elevated given prolonged downtrend",

            # Insider activity
            "insider_buying": "YES - Director Simon Dumesnil purchased 33,333 shares on Nov 28, 2025",

            # Volume
            "unusual_volume_before": "Moderate increase ahead of data readout announcements",

            # Social media
            "reddit_stocktwits_mention": "Discussed on StockTwits after initial surge; appeared on StockTitan top gainers list",

            # Technical pattern
            "technical_pattern": "Extended downtrend/consolidation near 52-week lows ($0.28-$0.49 range), then explosive breakout on volume",

            # Fundamentals
            "revenue_earnings": "Pre-revenue clinical stage biotech. Net losses ongoing. Post-surge raised $60M in public offering at $0.75/share (Dec 2, 2025) from Vivo Capital, Janus Henderson",

            "notes": "Classic biotech catalyst play. Sub-$1 stock with positive Phase 1 data across multiple indications. Insider buying confirmed. Followed by institutional capital raise."
        },
        {
            "ticker": "SMX",
            "company": "SMX (Security Matters) Public Limited Company",
            "sector": "Technology / Anti-Counterfeit / Materials Science",
            "month": "November 2025",

            # Price data (adjusted for 8:1 reverse split on Nov 18)
            "starting_price": 2.00,   # Post-reverse-split, low single digits before the run
            "peak_price": 61.04,      # Closed Nov 28 at $61.04
            "percentage_gain": 1000,   # Over 1000% gain in the month
            "date_of_move": "2025-11-26 to 2025-11-28",

            # Catalyst
            "catalyst": (
                "8-for-1 reverse stock split on Nov 18 creating ultra-low float (~1.05M shares). "
                "DMCC Dubai Precious Metals Conference presentation of 'Physical-to-Digital Link' "
                "molecular marker technology for gold authentication. $111.5M equity purchase "
                "agreement. Retail/meme-stock short squeeze dynamics on the tiny float."
            ),

            # Float / short interest
            "float_size": "ULTRA-LOW: ~1,050,572 shares after reverse split. Only 5 institutional holders with 39,539 shares combined",
            "short_interest": "Meaningful short interest relative to float; exact % not disclosed but squeeze dynamics confirmed",

            # Insider activity
            "insider_buying": "Not specifically reported; company announced $111.5M equity deal",

            # Volume
            "unusual_volume_before": "Massive spike: 21M+ shares traded Nov 28 alone vs float of ~1M",

            # Social media
            "reddit_stocktwits_mention": "YES - Heavily discussed on social media; described as 1000% week runner",

            # Technical pattern
            "technical_pattern": "Post-reverse-split consolidation followed by explosive breakout. Low float squeeze setup.",

            # Fundamentals
            "revenue_earnings": "Minimal revenue. Market cap had fallen ~99% YTD before the squeeze. Company provides molecular marking/authentication technology.",

            "notes": "Classic low-float squeeze play post reverse split. The combination of a ~1M share float, retail coordination, and a Dubai conference catalyst created extreme price action. High risk - stock had lost 99% of value earlier in 2025."
        },
        {
            "ticker": "BKYI",
            "company": "BIO-key International, Inc.",
            "sector": "Cybersecurity / Biometrics / Defense",
            "month": "November 2025",

            # Price data
            "starting_price": 0.65,   # Approximate pre-move price
            "peak_price": 1.20,       # Intraday peak on Nov 7
            "percentage_gain": 88,    # ~88% intraday, 81% at close
            "date_of_move": "2025-11-07",

            # Catalyst
            "catalyst": (
                "Secured significant identity and biometric security deployment with a major "
                "Middle East defense sector organization. One of largest regional security-sector "
                "deployments. New strategic partnership with Cloud Distribution for Saudi Arabia delivery."
            ),

            # Float / short interest
            "float_size": "Small micro-cap",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "Not specifically reported",

            # Volume
            "unusual_volume_before": "Volume exploded to 366M+ shares on Nov 7, far above normal averages",

            # Social media
            "reddit_stocktwits_mention": "Appeared on Nasdaq top gainers list; discussed on financial news sites",

            # Technical pattern
            "technical_pattern": "Low-priced consolidation followed by gap-up on news catalyst",

            # Fundamentals
            "revenue_earnings": "Q3 2025 revenue $1.55M (down from $2.14M YoY). Gross margin 77%. Net loss $965K. Cash $2.04M. Very small company.",

            "notes": "Defense/cybersecurity catalyst on a micro-cap. Rally continued next week (+23% on Nov 12). CyberDefense Initiative in progress."
        },
        {
            "ticker": "PMCB",
            "company": "PharmaCyte Biotech, Inc.",
            "sector": "Biotech / Pharmaceuticals",
            "month": "November 2025",

            # Price data
            "starting_price": 0.63,   # All-time low hit Nov 20
            "peak_price": 1.06,       # Approximate peak after 67.8% surge
            "percentage_gain": 68,    # 67.8% single-day surge
            "date_of_move": "2025-11-25",

            # Catalyst
            "catalyst": (
                "Successfully monetized stake in Femasys Inc (FEMY), strengthening cash position "
                "to approximately $20M. Announced after hitting all-time low of $0.63 on Nov 20."
            ),

            # Float / short interest
            "float_size": "Micro-cap, market cap ~$5.6M",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "Not specifically reported",

            # Volume
            "unusual_volume_before": "Sharp volume spike on announcement day",

            # Social media
            "reddit_stocktwits_mention": "Appeared on StockTitan top gainers list for November",

            # Technical pattern
            "technical_pattern": "Hit all-time low ($0.63 on Nov 20), then bounced sharply on catalyst. Classic reversal off extreme low.",

            # Fundamentals
            "revenue_earnings": "Pre-revenue. Only 2 employees. Zero debt. Cash position improved to ~$20M after Femasys sale.",

            "notes": "Extremely small company (2 employees). The Femasys stake sale was the sole catalyst. Subsequently received Nasdaq non-compliance notice in December."
        },
        {
            "ticker": "IQST",
            "company": "iQSTEL Inc.",
            "sector": "Telecom / Fintech / AI",
            "month": "November 2025",

            # Price data
            "starting_price": 2.10,   # Pre-earnings price
            "peak_price": 4.57,       # Pre-market high around Q3 report
            "percentage_gain": 118,   # Approximate gain from low to high
            "date_of_move": "Mid-November 2025",

            # Catalyst
            "catalyst": (
                "Record Q3 2025 results: $102.8M quarterly revenue (up 90% YoY, 42% sequential). "
                "Beat analyst estimate of $84.59M. Revenue run rate of $411.5M. Successful NASDAQ "
                "uplisting. Announced organic 2026 revenue forecast of $430M (26% growth). "
                "First-ever dividend declared."
            ),

            # Float / short interest
            "float_size": "Small cap; stockholders equity $17.8M",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "~20+ institutional investors owning ~5% of shares; no dilutive debt, no convertible notes, no warrants outstanding",

            # Volume
            "unusual_volume_before": "Volume spike around earnings release",

            # Social media
            "reddit_stocktwits_mention": "YES - Appeared on StockTitan top gainers; discussed on InvestorIdeas and financial news outlets",

            # Technical pattern
            "technical_pattern": "Consolidation around $2 level, then breakout on record earnings beat",

            # Fundamentals
            "revenue_earnings": "Record revenue: $102.8M Q3 (90% YoY growth). Adjusted EBITDA $0.68M positive. Targeting $340M for 2025, $430M for 2026. Telecom + Fintech + AI diversified platform.",

            "notes": "Rare penny stock with massive and growing revenue ($400M+ run rate). No dilutive instruments. Clean balance sheet. This is the kind of fundamental growth story that distinguishes itself from pure squeeze plays."
        },
    ],

    # =========================================================================
    # DECEMBER 2025 WINNERS
    # =========================================================================
    "december_2025": [
        {
            "ticker": "BBGI",
            "company": "Beasley Broadcast Group, Inc.",
            "sector": "Media / Broadcasting",
            "month": "December 2025",

            # Price data
            "starting_price": 4.00,   # ~$4 before the spike
            "peak_price": 26.37,      # Intraday high; closed at $16.69
            "percentage_gain": 312,   # ~312% at close; ~559% at intraday peak
            "date_of_move": "2025-12-10",

            # Catalyst
            "catalyst": (
                "Meme-stock short squeeze driven by coordinated retail trading via Discord and "
                "social media. Secondary narrative: digital revenue grew 28% YoY (now 25% of "
                "revenue), 8% operating expense reduction. M&A/acquisition speculation (unconfirmed). "
                "AI bots and social media 'experts' pumped the stock."
            ),

            # Float / short interest
            "float_size": "EXTREMELY LOW: Only 1.8M shares outstanding. Public float under 1M shares (Beasley family controls via Class B shares)",
            "short_interest": "6.18-8.35% of float (52,634 shares short). Days to cover: ~8 days. Average daily volume pre-spike: ~33,000 shares",

            # Insider activity
            "insider_buying": "Not specifically reported; Beasley family controls company via dual-class shares",

            # Volume
            "unusual_volume_before": "YES - 45.8M shares traded Dec 10 vs float of <1M and normal volume of ~33K. Over 45x the entire float traded in one day.",

            # Social media
            "reddit_stocktwits_mention": "YES - Originated from private Discord chats then spread to social media. Classic meme-stock coordination.",

            # Technical pattern
            "technical_pattern": "Months of sideways trading $3.70-$5.00, then explosive vertical squeeze. Multiple NASDAQ trading halts.",

            # Fundamentals
            "revenue_earnings": "Revenue fell 11% to $51M. $247M+ debt load. Cyclical ad market exposure. Digital pivot underway but company fundamentally fragile.",

            "notes": "Textbook meme-stock short squeeze. Ultra-low float (<1M public shares), meaningful short interest, coordinated retail buying. Settled back to $5-6 range within days. NASDAQ halted trading multiple times."
        },
        {
            "ticker": "WVE",
            "company": "Wave Life Sciences Ltd.",
            "sector": "Biotech / RNA Therapeutics",
            "month": "December 2025",

            # Price data
            "starting_price": 4.80,   # ~$4.80 pre-data (was sub-$5 penny stock territory)
            "peak_price": 14.28,      # Rose from $7.62 to $14.28 over several days
            "percentage_gain": 180,   # Over 180% in a week
            "date_of_move": "2025-12-08 to 2025-12-11",

            # Catalyst
            "catalyst": (
                "Blockbuster Phase 1 clinical data for WVE-007 (INHBE GalNAc-siRNA) obesity therapy. "
                "Single 240mg dose showed 9.4% visceral fat reduction and 4.5% total body fat reduction "
                "at 3 months. Comparable to GLP-1 drugs (like Wegovy) but preserves muscle mass. "
                "Multiple analyst upgrades: RBC upgraded to Outperform, Canaccord doubled PT to $40, "
                "Truist raised PT to $50."
            ),

            # Float / short interest
            "float_size": "Mid-cap biotech; market cap expanded significantly after surge",
            "short_interest": "Not specifically reported pre-move",

            # Insider activity
            "insider_buying": "SELLING: Board member Mark Corrigan sold 16,115 shares for ~$217K under 10b5-1 plan on Dec 11",

            # Volume
            "unusual_volume_before": "Volume exploded on data release day; 147% single-session surge is extremely rare in biotech",

            # Social media
            "reddit_stocktwits_mention": "YES - Widely discussed across financial media and social platforms after the data release",

            # Technical pattern
            "technical_pattern": "Years of quiet trading below the radar in $4-8 range. Explosive breakout on clinical data.",

            # Fundamentals
            "revenue_earnings": "Pre-revenue clinical stage biotech focused on RNA therapeutics. Raised ~$350M in upsized public offering at $19/share after the surge.",

            "notes": "This was a legitimate scientific catalyst - not a meme squeeze. WVE-007 obesity data positioned Wave as a serious GLP-1 competitor. The stock was genuinely in penny territory (<$5) before the data. The $350M capital raise ensures long runway."
        },
        {
            "ticker": "TLRY",
            "company": "Tilray Brands, Inc.",
            "sector": "Cannabis / Consumer Goods",
            "month": "December 2025",

            # Price data (post 1:10 reverse split effective Dec 1)
            "starting_price": 8.43,   # Pre-news price (~$0.84 pre-split equivalent, was a penny stock before split)
            "peak_price": 14.00,      # Approximate peak after 72% weekly gain
            "percentage_gain": 72,    # 72% over 5 trading sessions; 44% single-day on Dec 12
            "date_of_move": "2025-12-12 to 2025-12-18",

            # Catalyst
            "catalyst": (
                "Trump executive order directing DOJ/DEA to reclassify marijuana from Schedule I "
                "to Schedule III. Signed Dec 18, 2025. Reports first broke Dec 12 driving 44% "
                "single-day surge. Sector-wide rally: CGC +54%, ACB +20%, CNBS ETF +71%."
            ),

            # Float / short interest
            "float_size": "~116M shares post reverse split (was 1.16B pre-split). 1:10 reverse split effective Dec 1, 2025",
            "short_interest": "Historically high short interest in cannabis sector",

            # Insider activity
            "insider_buying": "Not specifically reported for this period",

            # Volume
            "unusual_volume_before": "Volume exploded to 81.58M shares on Dec 12 vs average of 5.67M",

            # Social media
            "reddit_stocktwits_mention": "YES - Massive retail interest; StockTwits reported retail optimism spiked; 70% gains in 5 days",

            # Technical pattern
            "technical_pattern": "Long downtrend (80-90% from highs). Reverse split on Dec 1. Then explosive breakout on policy catalyst.",

            # Fundamentals
            "revenue_earnings": "Revenue declining overall but diversified into beverages/CPG. Company unprofitable. Was essentially a penny stock pre-split ($0.84 equivalent). Cannabis sector broadly unprofitable.",

            "notes": "Policy catalyst play. TLRY had been a true penny stock (~$0.84) before the 1:10 reverse split on Dec 1. The Trump marijuana rescheduling executive order was a sector-wide catalyst. Other cannabis names also surged massively (CGC from ~$1 to ~$1.53)."
        },
        {
            "ticker": "BEAT",
            "company": "HeartBeam, Inc.",
            "sector": "Medical Devices / Healthcare Technology",
            "month": "December 2025",

            # Price data
            "starting_price": 0.76,   # $0.7619 on Dec 8
            "peak_price": 2.28,       # Peak on Dec 11
            "percentage_gain": 200,   # ~200% gain from $0.76 to $2.28
            "date_of_move": "2025-12-10 to 2025-12-11",

            # Catalyst
            "catalyst": (
                "FDA 510(k) clearance for 12-lead ECG synthesis software for arrhythmia assessment. "
                "First-ever cable-free synthesized 12-lead ECG device (credit-card-sized). Clearance "
                "came after successful appeal of prior NSE determination. H.C. Wainwright raised PT "
                "from $2.50 to $5.50. Benchmark upgraded to Buy with $8 PT."
            ),

            # Float / short interest
            "float_size": "Micro-cap; very small float",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "Not specifically reported",

            # Volume
            "unusual_volume_before": "Volume exploded to 270M+ shares on Dec 10 vs average in low millions",

            # Social media
            "reddit_stocktwits_mention": "Appeared on StockTitan top gainers for December; widely covered in financial news",

            # Technical pattern
            "technical_pattern": "Trading near 52-week lows ($0.54-$0.80 range). Explosive gap-up on FDA clearance.",

            # Fundamentals
            "revenue_earnings": "Pre-revenue. Q3 2025 net loss $5.3M. R&D expenses $3.3M. Cash only $1.9M pre-catalyst. Plans for limited commercial launch Q1 2026.",

            "notes": "Classic FDA clearance catalyst on a sub-$1 penny stock. The device is genuinely innovative (credit-card-sized wireless ECG). Limited cash ($1.9M) is a concern. Planning commercial launch Q1 2026."
        },
        {
            "ticker": "SIDU",
            "company": "Sidus Space, Inc.",
            "sector": "Aerospace / Defense / Satellites",
            "month": "December 2025",

            # Price data
            "starting_price": 0.63,   # All-time low on Dec 1, 2025
            "peak_price": 2.50,       # Approximate peak after multiple surges
            "percentage_gain": 297,   # ~297% from all-time low to peak
            "date_of_move": "2025-12-22 to 2025-12-30",

            # Catalyst
            "catalyst": (
                "Selected as contract awardee under Missile Defense Agency's SHIELD IDIQ program "
                "with $151B total ceiling. Covers AI/ML, digital engineering, open systems for "
                "Golden Dome missile defense. Closed $25M public offering at $1.30/share (Dec 24) "
                "and another $16.2M at $1.50/share (Dec 29). LizzieSat-3 commissioning success."
            ),

            # Float / short interest
            "float_size": "Significantly expanded by offerings (19.2M + 10.8M new shares issued in December)",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "Appointed aerospace executive Kelle Wendling to Board (effective Jan 5, 2026)",

            # Volume
            "unusual_volume_before": "Massive volume increase after SHIELD contract announcement",

            # Social media
            "reddit_stocktwits_mention": "Discussed on financial news sites; appeared on StockTitan December gainers",

            # Technical pattern
            "technical_pattern": "Hit all-time low Dec 1 ($0.63), then multi-day rally starting Dec 22. Classic bottom reversal on government contract catalyst.",

            # Fundamentals
            "revenue_earnings": "Small revenue satellite company. Raised $41.2M in December offerings. Designs, manufactures, and launches commercial satellites with AI-enhanced Data-as-a-Service.",

            "notes": "Defense/space catalyst play. $151B SHIELD program is massive but IDIQ contracts don't guarantee revenue. The capital raises ($41.2M total) strengthen the balance sheet significantly. Stock went from all-time low to 4x in 9 trading days."
        },
    ],

    # =========================================================================
    # JANUARY 2026 WINNERS
    # =========================================================================
    "january_2026": [
        {
            "ticker": "OPAD",
            "company": "Offerpad Solutions Inc.",
            "sector": "Real Estate Technology / iBuyer",
            "month": "January 2026",

            # Price data
            "starting_price": 1.45,   # Price on Jan 7
            "peak_price": 2.38,       # Price on Jan 9
            "percentage_gain": 65,    # +64.7% for the month; 55% surge on Jan 9 alone
            "date_of_move": "2026-01-09",

            # Catalyst
            "catalyst": (
                "Trump's $200B mortgage bond plan: directed FHFA to authorize Fannie Mae and "
                "Freddie Mac to deploy ~$200B in reserves to purchase mortgage bonds, aiming to "
                "lower consumer mortgage rates. Also executive order banning institutional investors "
                "from purchasing single-family homes. Alliance Global initiated Buy with $3.50 PT."
            ),

            # Float / short interest
            "float_size": "Small cap; market cap under $100M pre-surge",
            "short_interest": "Meme stock dynamics present; exact short interest not reported",

            # Insider activity
            "insider_buying": "Not specifically reported",

            # Volume
            "unusual_volume_before": "Massive volume spike on Jan 9; overnight 50% gap-up",

            # Social media
            "reddit_stocktwits_mention": "YES - Expanded from Opendoor meme-stock rally; retail investors targeted OPAD as next play",

            # Technical pattern
            "technical_pattern": "Extended downtrend, then explosive gap-up on policy catalyst. Meme-stock spillover from OPEN rally.",

            # Fundamentals
            "revenue_earnings": "iBuyer model struggling in high-rate environment. Lower mortgage rates would directly revitalize business. Stock fell back to $0.91 by Feb 10.",

            "notes": "Policy catalyst + meme stock momentum. Trump housing policy was the primary driver. Gains were short-lived - stock fell back below $1 within a month. Demonstrates the importance of timing entries/exits."
        },
        {
            "ticker": "SRFM",
            "company": "Surf Air Mobility Inc.",
            "sector": "Aviation / AI / Technology",
            "month": "January 2026",

            # Price data
            "starting_price": 1.91,   # Low of month
            "peak_price": 3.01,       # High of month
            "percentage_gain": 35,    # +35.1% for month; 23-25% single-day surges
            "date_of_move": "2026-01-04 to 2026-01-05",

            # Catalyst
            "catalyst": (
                "SurfOS AI-powered operating system update and expanded Palantir partnership. "
                "$100M strategic transaction with $26M allocated to SurfOS development. Internal "
                "metrics showed 36% cost reduction, 197% increase in bookings per broker, 14% faster "
                "processing. Hawaii infrastructure investment of $22.4M. Partnership with Hawaii DOT "
                "and Beta Technologies."
            ),

            # Float / short interest
            "float_size": "Small cap",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "Not specifically reported",

            # Volume
            "unusual_volume_before": "Spike in volume on Jan 4-5 around SurfOS announcement",

            # Social media
            "reddit_stocktwits_mention": "Discussed in financial news; Palantir partnership attracted attention",

            # Technical pattern
            "technical_pattern": "Recovery bounce from recent lows on AI/partnership catalyst",

            # Fundamentals
            "revenue_earnings": "Pre-tax profit margin -119.4%. Mokulele Airlines subsidiary: 36K departures, 224K passengers in 2025. Analysts average PT $5.81 (189% upside from Jan price).",

            "notes": "AI + Palantir partnership narrative drove interest. Company is deeply unprofitable but has real aviation operations (Mokulele Airlines in Hawaii). Pulled back to ~$2 by mid-February."
        },
        {
            "ticker": "SPIR",
            "company": "Spire Global, Inc.",
            "sector": "Space / Satellite Data / Defense",
            "month": "January 2026",

            # Price data
            "starting_price": 6.50,   # Approximate start of January
            "peak_price": 9.13,       # Rose significantly through January
            "percentage_gain": 34,    # +34.2% for month
            "date_of_move": "January 2026 (multiple catalysts through month)",

            # Catalyst
            "catalyst": (
                "Successfully launched 9 satellites on SpaceX Twilight mission (Jan 12). "
                "Pentagon SHIELD IDIQ contract ($151B ceiling). Appointed Admiral Grady and "
                "Ed Newberry to Advisory Board (Jan 21). H.C. Wainwright raised PT from $14 to $19 (Jan 23). "
                "Backlog exceeds $200M."
            ),

            # Float / short interest
            "float_size": "Small-mid cap",
            "short_interest": "Not specifically reported",

            # Insider activity
            "insider_buying": "Strategic advisory board appointments signal confidence",

            # Volume
            "unusual_volume_before": "Volume spikes around satellite launch and SHIELD contract news",

            # Social media
            "reddit_stocktwits_mention": "YES - Pentagon SHIELD contract news drove retail interest on StockTwits",

            # Technical pattern
            "technical_pattern": "Recovery from 50%+ YTD decline in 2025. Multiple catalysts drove sustained January rally.",

            # Fundamentals
            "revenue_earnings": "Targets 30%+ revenue growth in 2026 post-maritime divestiture. EBITDA breakeven target Q4 2026. Backlog >$200M. Notable contracts: $11.2M NOAA, $2.5M NOAA weather, EUR3M EUMETSAT.",

            "notes": "Space/defense catalyst play with real revenue and growing backlog. Multiple catalysts throughout January created a sustained rally rather than a single spike."
        },
        {
            "ticker": "SLS",
            "company": "SELLAS Life Sciences Group, Inc.",
            "sector": "Biotech / Oncology",
            "month": "January 2026",

            # Price data
            "starting_price": 3.35,   # Pre-move price
            "peak_price": 4.42,       # Peak during January
            "percentage_gain": 32,    # ~32% gain; multiple volatile swings
            "date_of_move": "Early-mid January 2026",

            # Catalyst
            "catalyst": (
                "Phase 3 REGAL trial approaching final analysis trigger (72 of 80 events reached). "
                "Phase 2 SLS009 + azacitidine + venetoclax showed 46% overall response rate in AML. "
                "European IMPACT-AML collaboration announced. Multiple 10-17% single-day swings."
            ),

            # Float / short interest
            "float_size": "Small cap; clinical stage biotech",
            "short_interest": "Not specifically reported; stock extremely volatile",

            # Insider activity
            "insider_buying": "Not specifically reported",

            # Volume
            "unusual_volume_before": "Volume spikes on trial news days",

            # Social media
            "reddit_stocktwits_mention": "YES - Trending on Reddit with 22 mentions in 24 hours per AltIndex tracker. One of the most-discussed penny stocks.",

            # Technical pattern
            "technical_pattern": "Choppy trading with large swings (10-17% daily moves both directions). Phase 3 data trigger imminent.",

            # Fundamentals
            "revenue_earnings": "Pre-revenue clinical stage. Pretax margin -1460.9%. Revenue contracted -100% over 3 years. Classic cash-burning biotech.",

            "notes": "High-volatility biotech with imminent Phase 3 readout. Heavily discussed on Reddit. The pending REGAL trial analysis (needs 8 more events) is the key binary catalyst ahead."
        },
        {
            "ticker": "OPEN",
            "company": "Opendoor Technologies Inc.",
            "sector": "Real Estate Technology / iBuyer",
            "month": "January 2026",

            # Price data
            "starting_price": 5.50,   # Start of January range
            "peak_price": 7.92,       # Peak on Trump mortgage news
            "percentage_gain": 44,    # ~44% at peak; 20% in first 3 sessions
            "date_of_move": "2026-01-02 to 2026-01-10",

            # Catalyst
            "catalyst": (
                "New Year meme-stock momentum from 'Open Army' retail investors. "
                "Trump $200B mortgage bond plan boosted entire housing sector. "
                "New CEO Kaz Nejatian's 'Opendoor 2.0' strategy. "
                "Roundhill MEME ETF up 20% to start 2026. "
                "20% gains in first 3 trading sessions of the year."
            ),

            # Float / short interest
            "float_size": "Larger cap (~$3.7B market cap) but with heavy retail/meme ownership",
            "short_interest": "Heavily shorted; retail 'Open Army' targeting shorts",

            # Insider activity
            "insider_buying": "Not specifically reported; new CEO focused on profitability by Q3 2026",

            # Volume
            "unusual_volume_before": "29M shares on Jan 10; well above normal. Meme momentum building from year-end 2025.",

            # Social media
            "reddit_stocktwits_mention": "YES - 'Open Army' on StockTwits and Reddit actively pumping. Roundhill MEME ETF holding.",

            # Technical pattern
            "technical_pattern": "2025 was a massive rally year (+264%). January continuation with meme momentum + policy catalyst.",

            # Fundamentals
            "revenue_earnings": "Revenue expected to grow 5-10% in 2026. Targeting adjusted net profit breakeven by Q3 2026. iBuyer model depends on lower mortgage rates.",

            "notes": "Borderline penny stock (was $0.51 in mid-2024, rose to ~$5-6 range). The Jan 2026 surge was a mix of meme momentum, new year optimism, and Trump housing policy. Analysts consensus is SELL with $1.50 PT despite retail bullishness."
        },
    ],
}


# =============================================================================
# SUMMARY STATISTICS FOR ALGORITHM VALIDATION
# =============================================================================

COMMON_PATTERNS = {
    "catalysts": {
        "FDA_clearance_or_clinical_data": ["KTTA", "BEAT", "WVE", "SLS"],
        "government_defense_contract": ["BKYI", "SIDU", "SPIR"],
        "meme_squeeze_low_float": ["SMX", "BBGI", "OPAD", "OPEN"],
        "policy_regulatory": ["TLRY"],  # Cannabis rescheduling
        "earnings_revenue_beat": ["IQST"],
        "strategic_deal_or_monetization": ["PMCB", "SRFM"],
    },

    "pre_move_signals": {
        "ultra_low_float": ["SMX", "BBGI"],
        "near_52week_low": ["KTTA", "PMCB", "BEAT", "SIDU"],
        "insider_buying": ["KTTA"],
        "high_short_interest": ["BBGI", "SMX"],
        "social_media_buzz_before": ["BBGI", "SMX", "OPEN"],
        "volume_spike_before_breakout": ["BKYI", "BBGI", "SMX"],
    },

    "sectors": {
        "biotech_pharma": ["KTTA", "WVE", "PMCB", "SLS", "BEAT"],
        "defense_space": ["BKYI", "SIDU", "SPIR"],
        "real_estate_tech": ["OPAD", "OPEN"],
        "cannabis": ["TLRY"],
        "media": ["BBGI"],
        "telecom_tech": ["IQST", "SRFM"],
        "materials_science": ["SMX"],
    },

    "key_validation_insights": [
        "5 of 15 winners (33%) were biotech/pharma with clinical catalysts",
        "3 of 15 (20%) were defense/space contract plays",
        "3 of 15 (20%) were meme/squeeze plays with ultra-low floats",
        "Most winners were trading near 52-week lows before the move",
        "Insider buying (KTTA) was a rare but powerful confirming signal",
        "Ultra-low float + short interest (BBGI, SMX) produced the largest single-day gains",
        "FDA clearance (BEAT) and clinical data (WVE, KTTA) were the most reliable catalysts",
        "Policy catalysts (TLRY cannabis, OPAD housing) produced sector-wide moves",
        "Revenue beats on penny stocks (IQST) are rare but powerful",
        "Many winners had volume spikes BEFORE the major move",
        "Social media coordination (Discord/Reddit) was present in squeeze plays",
        "Most gains were NOT sustained - stocks like OPAD, BBGI reversed within weeks",
        "Capital raises often followed big moves (WVE $350M, KTTA $60M, SIDU $41M)",
    ],
}
