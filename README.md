# 🚀 IntelliTrail: AI-Powered Multi-Agent Risk Management Engine

**IntelliTrail** is an advanced algorithmic trading terminal designed to replace inefficient, static stop-losses with a dynamic, machine-learning-driven risk engine. Built for the **Nifty 100** constituents, it manages multiple "Agents" (Bots) that adapt to real-time market volatility to maximize profit retention and optimize entry points.


# 📖 Project Overview

Traditional trading strategies often use a "dumb" fixed-percentage stop loss (e.g., 5%), which fails to account for market noise or momentum.  

**IntelliTrail** solves this by using a **Linear Regression** model to predict price trends and adjust the **Trailing Stop-Loss (TSLO)** dynamically based on the **Average True Range (ATR)**.


# 🛠️ Core Features & Implementation

## 1. The AI "Brain"

- **Model:** Linear Regression trained on high-frequency intraday data  
- **Feature Engineering:** Uses  
  - SMA (20/50)  
  - RSI (Momentum)  
  - ATR (Volatility)  

- **Dynamic Multiplier:**  
  - **1.2 × ATR** during bearish predictions  
  - **2.0 × ATR** during bullish trends  

This allows the stop-loss to tighten during risk conditions and loosen during strong trends.


## 2. Multi-Agent Bot System

### Auto-Protect (Sell Bots)
Protects existing holdings by trailing the stop-loss upward, ensuring the exit occurs at the highest possible point before a trend reversal.

### Auto-Scout (Buy Bots)
Designed to find the **Minimum Price of the day**.

Entry target formula:
```bash
Entry Target = Daily Low − (ATR × 0.5)
```

## 3. Interactive Command Center (Dashboard)

- **Real-time Monitoring** built using **Streamlit**
- **Strategy Comparison:** AI vs Manual execution comparison with visual charts
- **Persistent Storage:** Uses **Supabase** database to track all orders and executions
- **Fleet Management:** Centralized **Stats Dashboard** showing system health and total managed shares
- **Execution Tracking:** Step-by-step logs for every bot deployment


# 📁 Project Structure
```
IntelliTrail/
│
├── core/                          # Backend Logic
│   ├── __init__.py                # Package marker
│   ├── data_loader.py             # Ingests stock data from CSVs
│   ├── indicators.py              # Calculates RSI, ATR, SMA, BB
│   ├── engine.py                  # TSLO and Buy-Scout algorithms
│   └── bot_orchestrator.py        # AI Bot execution pipeline
│
├── dashboard/                     # User Interface
│   └── app.py                     # Streamlit Command Center
│
├── models/                        # Trained ML Models
│   ├── train_model.py             # Model training script
│   └── trained_models/            # Saved .pkl files
│
├── data/raw/                      # Stock datasets
│
├── test_bot_execution.py          # System testing script
├── BOT_ARCHITECTURE.md            # Technical documentation
├── QUICKSTART.md                  # Setup guide
└── main.py                        # CLI simulation script
```


# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/PoorvaJawale/IntelliTrail-AI-Trading-Bot.git
cd IntelliTrail
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Set Up Environment Variables

Create a `.env` file with your Supabase credentials:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

The database schema will be automatically created on first connection.

## 4️⃣ Train the AI Model (if needed)

```bash
python main.py
```

## 5️⃣ Test the System

```bash
python test_bot_execution.py
```

## 6️⃣ Run the Dashboard

```bash
streamlit run dashboard/app.py
```

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md)

# 📊 Performance Metrics
The system evaluates performance using Profit Retention %.

This metric measures:
```bash
Profit Retention % =
(Profit Captured by Bot / Peak Possible Profit) × 100
```
It compares the AI dynamic trailing strategy against a traditional static 5% stop-loss baseline.

# Developed By
Poorva Jawale

# ⚙️ Tech Stack

- **Python** - Core programming language
- **Scikit-Learn** - Machine learning (Linear Regression)
- **Pandas** - Data manipulation and analysis
- **Pandas_TA** - Technical indicators (SMA, RSI, ATR, BB)
- **Streamlit** - Interactive web dashboard
- **Plotly** - Advanced charting and visualizations
- **Supabase** - PostgreSQL database for persistent storage
- **Joblib** - Model serialization

# 📚 Documentation

- [BOT_ARCHITECTURE.md](BOT_ARCHITECTURE.md) - Complete technical architecture
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step setup guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - What was built and how it works

# 🎯 Key Features

- **Persistent Data Storage**: All orders and executions stored in Supabase
- **AI-Driven Execution**: Dynamic price optimization using ML
- **Real-Time Tracking**: Step-by-step execution logs
- **Multi-Bot Management**: Deploy and manage multiple bots simultaneously
- **AI vs Manual Comparison**: Visual proof of AI superiority
- **Complete Audit Trail**: Every decision logged and traceable
- **Scalable Architecture**: Ready for production deployment
