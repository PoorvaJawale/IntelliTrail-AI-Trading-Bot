from core.data_loader import load_stock_data
from core.indicators import apply_indicators
from models.train_model import train_and_save_model
from core.engine import run_simulation_with_ai
from core.comparison import calculate_metrics
from core.engine import run_simulation # The non-AI version we made earlier

# 1. Load and Prep
df = load_stock_data("data/raw/NIFTY 50_daily.csv")
df = apply_indicators(df)

# 2. Train AI
ai_model = train_and_save_model(df)

# 3. Run AI-Powered Simulation
start_index = -60
test_df = df.iloc[start_index:]
entry_price = test_df.iloc[0]['close']

run_simulation_with_ai(test_df, entry_price, ai_model)

# 1. Run Static 2% Simulation
static_trades, _ = run_simulation(test_df, entry_price, risk_multiplier=2.0)
static_results = calculate_metrics(test_df, static_trades, entry_price)

# 2. Run IntelliTrail (AI) Simulation
ai_trades = run_simulation_with_ai(test_df, entry_price, ai_model)
ai_results = calculate_metrics(test_df, ai_trades, entry_price)

print("\n--- BATTLE OF THE STRATEGIES ---")
print(f"Static 2% Strategy: {static_results}")
print(f"IntelliTrail AI:    {ai_results}")