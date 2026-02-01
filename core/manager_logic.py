def process_portfolio(active_bots, model):
    """
    Simulates real-time processing for multiple assets.
    """
    portfolio_results = {}
    for ticker, config in active_bots.items():
        # 1. Load the specific minute-wise file
        file_path = f"data/raw/{ticker}_minute.csv"
        df = load_stock_data(file_path)
        df = apply_indicators(df)
        
        # 2. Run the specific strategy
        if config['type'] == "SELL (Protect)":
            trades, history = run_simulation_with_ai(df, config['trigger'], model)
        else:
            trades, history = run_buy_simulation(df, config['trigger'])
            
        portfolio_results[ticker] = {
            "trades": trades,
            "history": history,
            "current_price": df.iloc[-1]['close']
        }
    return portfolio_results