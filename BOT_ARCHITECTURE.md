# IntelliTrail AI Bot Architecture

## Overview

The IntelliTrail AI Trading Bot system is a complete end-to-end algorithmic trading platform that uses machine learning to optimize buy and sell execution prices.

## System Components

### 1. Database Layer (Supabase)

#### Tables

**orders**
- Stores all trading orders (buy/sell) with their execution details
- Tracks order lifecycle: pending -> executed/cancelled/failed
- Key fields: ticker, strategy_type, quantity, ai_executed_price, pnl, status

**bot_executions**
- Detailed step-by-step execution logs
- Tracks each stage: Data Loading, Technical Analysis, AI Execution
- Enables debugging and performance analysis

**ai_predictions**
- Stores AI model predictions for each order
- Includes confidence scores and feature data
- Enables model performance tracking

**manual_levels**
- Stores manual trading levels for comparison
- Examples: "2% dip", "5% gain", "10% gain"
- Used to demonstrate AI advantage over static rules

### 2. AI Bot Orchestrator

**Location:** `core/bot_orchestrator.py`

**Key Responsibilities:**
- Manages complete bot execution pipeline
- Loads historical data
- Applies technical indicators
- Runs AI predictions
- Calculates optimal execution prices
- Logs all steps to database

#### BUY Strategy Logic

The AI finds the minimum price using:

1. **Historical Low Detection**
   - Identifies absolute lowest price in simulation period

2. **ATR-Based Support Level**
   - Calculates: current_price - (ATR × 1.5)
   - Dynamic support based on volatility

3. **AI Price Prediction**
   - Uses trained Linear Regression model
   - Features: SMA_20, SMA_50, RSI, ATR
   - Applies 2% safety margin

4. **Bollinger Band Lower Bound**
   - Statistical support level
   - 2 standard deviations below mean

**Final Decision:** Takes the MINIMUM of all four signals

#### SELL Strategy Logic

The AI finds the maximum price using Dynamic Trailing Stop-Loss (TSLO):

1. **Initial Stop-Loss Setup**
   - Starts at: entry_price - (ATR × 2.0)

2. **AI-Driven Multiplier Adjustment**
   - Bearish AI prediction → 1.2× ATR (tight stop)
   - Bullish AI prediction → 2.0× ATR (loose stop)

3. **Trailing Mechanism**
   - Stop-loss only moves UP, never down
   - Locks in profits as price rises
   - Exits when price touches stop-loss

4. **Profit Maximization**
   - Tracks maximum profit achieved
   - Exits at optimal high point before reversal

### 3. Dashboard (Streamlit)

**Location:** `dashboard/app.py`

#### Tab 1: Stats
- Fleet performance overview
- Total bots, shares, P&L
- Execution log history
- Order status tracking

#### Tab 2: Analytics
- Technical analysis charts (Candlestick, SMA, Bollinger Bands)
- Technical indicators (RSI, ATR, Volatility)
- AI price predictions
- Performance metrics (Returns, Sharpe Ratio)

#### Tab 3: Portfolio
- Deploy new AI bots
- View active positions
- AI vs Manual comparison
- Execution logs
- Cancel orders

### 4. Data Pipeline

**Location:** `core/data_loader.py`, `core/indicators.py`

**Process:**
1. Load historical price data from CSV files
2. Apply technical indicators:
   - SMA (20, 50 period)
   - RSI (14 period)
   - ATR (14 period)
   - Bollinger Bands
3. Prepare features for AI model
4. Handle missing data and edge cases

### 5. AI Model

**Location:** `models/train_model.py`

**Algorithm:** Linear Regression

**Features:**
- SMA_20 (Short-term trend)
- SMA_50 (Long-term trend)
- RSI (Momentum indicator)
- ATR (Volatility measure)

**Target:** Next day's closing price

**Training:**
- 80/20 train-test split
- Chronological data ordering (no shuffle)
- Saved as pickle file for reuse

## Execution Flow

### Deploying a BUY Bot

1. User selects stock, quantity, sim_days
2. Bot Orchestrator creates pending order in database
3. Loads historical data for selected stock
4. Applies technical indicators
5. AI calculates optimal buy target using 4 signals
6. Simulates execution at historical minimum price
7. Stores results, predictions, manual levels in database
8. Updates order status to 'executed'
9. Calculates P&L vs current price
10. Dashboard displays results

### Deploying a SELL Bot

1. User selects stock, quantity, sim_days
2. Bot Orchestrator creates pending order in database
3. Loads historical data
4. Applies technical indicators
5. Initializes Dynamic TSLO
6. AI adjusts stop-loss multiplier at each timestep
7. Simulates exit at optimal high point
8. Stores results, TSLO history in database
9. Updates order status to 'executed'
10. Calculates P&L
11. Dashboard displays results

## AI vs Manual Comparison

For every order, the system calculates both:

**AI Execution:**
- Dynamic, data-driven decision
- Uses multiple signals
- Adapts to market conditions

**Manual Levels:**
- Static percentage-based rules
- Examples: "Buy at 2% dip", "Sell at 5% gain"
- No market intelligence

**Results Display:**
- Side-by-side price comparison
- Advantage calculation (AI price - Manual price)
- Visual bar chart
- Demonstrates AI superiority

## Performance Metrics

### Confidence Score
Calculated based on:
- Price deviation from target
- Market volatility (ATR)
- RSI momentum

Range: 0.0 to 1.0

### Profit & Loss (P&L)
- BUY: (current_price - executed_price) × quantity
- SELL: (executed_price - entry_price) × quantity

### Profit Retention (SELL only)
- Measures how much of peak profit was captured
- Formula: (actual_profit / peak_profit) × 100

## Database Security

All tables have Row Level Security (RLS) enabled:
- Public read access (for demo)
- Public insert/update/delete access (for demo)
- Can be restricted to authenticated users later

## Environment Variables

Required in `.env` file:
```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
```

## Testing

Run the test script:
```bash
python test_bot_execution.py
```

This will:
1. Test BUY order execution
2. Test SELL order execution
3. Fetch and display all orders from database
4. Verify complete pipeline

## Future Enhancements

1. Real-time market data integration
2. Live trading execution (broker API)
3. Multi-timeframe analysis (minute, hourly)
4. Advanced ML models (LSTM, Random Forest)
5. Risk management rules (position sizing, portfolio limits)
6. User authentication and multi-user support
7. Webhook notifications (email, SMS)
8. Backtesting framework
9. Performance analytics dashboard
10. Paper trading mode

## Key Advantages

1. **Data Persistence:** All orders stored in database
2. **Execution Tracking:** Step-by-step logs
3. **AI-Driven:** Adapts to market conditions
4. **Transparent:** Shows AI vs Manual comparison
5. **Scalable:** Can manage multiple bots simultaneously
6. **Auditable:** Complete execution history
7. **Flexible:** Supports both buy and sell strategies
