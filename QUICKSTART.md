# IntelliTrail AI Bot - Quick Start Guide

## Prerequisites

1. Python 3.8 or higher
2. Supabase account and project
3. Historical stock data in `data/raw/` folder

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

Get these credentials from your Supabase project dashboard:
- Go to Settings > API
- Copy the Project URL
- Copy the anon/public key

### 3. Database Setup

The database schema is automatically created using Supabase migrations.
The following tables will be created:
- orders
- bot_executions
- ai_predictions
- manual_levels

### 4. Train the AI Model (if not already done)

```bash
python main.py
```

This will:
- Load historical data
- Train the Linear Regression model
- Save the model to `models/trained_models/nifty_model.pkl`

### 5. Test the Bot System

```bash
python test_bot_execution.py
```

This will:
- Execute a test BUY order
- Execute a test SELL order
- Display results and verify database integration

### 6. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Using the Dashboard

### Stats Tab
- View all deployed bots
- Monitor fleet performance
- Check execution logs
- Track total P&L

### Analytics Tab
- Select any stock from dropdown
- View technical analysis charts
- See AI price predictions
- Analyze performance metrics

### Portfolio Tab

#### Deploy a New Bot

1. Select stock from dropdown
2. Choose strategy:
   - Auto-Scout (Buy): Finds minimum price
   - Auto-Protect (Sell): Maximizes exit price
3. Enter quantity (number of shares)
4. Set simulation days (5-60)
5. Click "DEPLOY AI BOT"

#### View Active Positions

- Each position shows:
  - AI execution price
  - Entry date and status
  - P&L calculation
  - AI vs Manual comparison
  - Execution logs

#### Cancel Orders

- Click "Cancel Order" button on any active position

## Understanding Results

### BUY Orders

The AI finds the minimum price using:
- Historical low detection
- ATR-based support levels
- AI price prediction
- Bollinger Band lower bound

The executed price will be the LOWEST of all signals.

### SELL Orders

The AI maximizes profit using:
- Dynamic Trailing Stop-Loss (TSLO)
- AI-adjusted risk multipliers
- Profit protection mechanism

The executed price will be at the OPTIMAL HIGH POINT.

### AI vs Manual Comparison

For every order, you'll see:
- AI Dynamic execution price
- Manual static levels (e.g., "2% dip", "5% gain")
- Price advantage of AI over each manual level
- Visual bar chart comparison

## Data Requirements

Your stock data files should be in `data/raw/` folder with this format:

```
RELIANCE_daily.csv
TCS_daily.csv
INFY_daily.csv
```

Each CSV should have these columns:
- date (or timestamp/datetime)
- open
- high
- low
- close
- volume (optional)

## Troubleshooting

### "Supabase credentials not found"
- Check that `.env` file exists in project root
- Verify SUPABASE_URL and SUPABASE_ANON_KEY are set correctly

### "No data found for [TICKER]"
- Ensure CSV file exists: `data/raw/[TICKER]_daily.csv`
- Check that file has correct column names (lowercase)

### "Model file not found"
- Run `python main.py` to train and save the model first

### Database Connection Issues
- Verify Supabase project is active
- Check that API keys are correct
- Ensure internet connection is stable

## Next Steps

1. Add more stock data files to `data/raw/`
2. Deploy multiple bots across different stocks
3. Monitor performance in Stats tab
4. Compare AI vs Manual execution results
5. Analyze technical indicators in Analytics tab

## Advanced Usage

### Custom Simulation Periods

Adjust `sim_days` to test different scenarios:
- 5-10 days: Short-term trading
- 20-30 days: Medium-term positions
- 40-60 days: Long-term analysis

### Multiple Timeframes

Add data files with different timeframes:
```
RELIANCE_daily.csv
RELIANCE_hourly.csv
RELIANCE_minute.csv
```

The Analytics tab will auto-detect available timeframes.

## Support

For issues or questions:
1. Check BOT_ARCHITECTURE.md for technical details
2. Review execution logs in Stats tab
3. Run test_bot_execution.py to verify system health
