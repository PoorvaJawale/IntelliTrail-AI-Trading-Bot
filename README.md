# 🚀 IntelliTrail: AI-Powered Multi-Agent Risk Management Engine

**IntelliTrail** is an advanced algorithmic trading terminal designed to replace inefficient, static stop-losses with a dynamic, machine-learning-driven risk engine. Built for the **Nifty 100** constituents, it manages multiple "Agents" (Bots) that adapt to real-time market volatility to maximize profit retention and optimize entry points.

---

# 📖 Project Overview

Traditional trading strategies often use a "dumb" fixed-percentage stop loss (e.g., 5%), which fails to account for market noise or momentum.  

**IntelliTrail** solves this by using a **Linear Regression** model to predict price trends and adjust the **Trailing Stop-Loss (TSLO)** dynamically based on the **Average True Range (ATR)**.

---

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

---

## 2. Multi-Agent Bot System

### Auto-Protect (Sell Bots)
Protects existing holdings by trailing the stop-loss upward, ensuring the exit occurs at the highest possible point before a trend reversal.

### Auto-Scout (Buy Bots)
Designed to find the **Minimum Price of the day**.

Entry target formula:
```bash
Entry Target = Daily Low − (ATR × 0.5)
```


---

## 3. Interactive Command Center (Dashboard)

- **Real-time Monitoring** built using **Streamlit**
- **Strategy Comparison:** "Battle of the Strategies" comparing AI vs static 5% stop-loss
- **Session Persistence:** Uses `st.session_state` to track active bots
- **Fleet Management:** Centralized **Stats Dashboard** showing system health and total managed shares

---

# 📁 Project Structure
IntelliTrail/
│
├── core/ # Backend Logic
│ ├── init.py # Package marker
│ ├── data_loader.py # Ingests Nifty 100 Kaggle CSVs
│ ├── indicators.py # Calculates RSI, ATR, SMA
│ └── engine.py # TSLO and Buy-Scout algorithms
│
├── dashboard/ # User Interface
│ └── app.py # Streamlit Command Center
│
├── models/ # Trained ML Models (.pkl files)
├── data/ # Raw Nifty 100 datasets
│
└── main.py # CLI simulation & strategy battle script


---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/PoorvaJawale/IntelliTrail-AI-Trading-Bot.git
cd IntelliTrail
```

## 2️⃣ Install Dependencies

```bash
pip install pandas pandas_ta scikit-learn streamlit joblib plotly
```

## 3️⃣ Run the Dashboard
```bash
streamlit run dashboard/app.py
```

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
Python
Scikit-Learn
Pandas
Pandas_TA
Streamlit
Plotly
