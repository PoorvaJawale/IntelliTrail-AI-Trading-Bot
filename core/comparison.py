def calculate_metrics(df, trades, entry_price):
    """
    Calculates key performance metrics for the presentation.
    """
    # SAFETY CHECK: If trades is accidentally a tuple (list, history), take only the list
    if isinstance(trades, tuple):
        trades = trades[0]

    if not trades or not isinstance(trades, list):
        return "No trades completed."

    try:
        exit_price = trades[0]['exit_price']
        final_profit = exit_price - entry_price
        
        # Peak profit during the trade
        max_price = df['close'].max()
        peak_profit = max_price - entry_price
        
        # Profit Retention: How much of the peak did we actually keep?
        retention = (final_profit / peak_profit) * 100 if peak_profit > 0 else 0
        
        return {
            "Final Profit": round(final_profit, 2),
            "Peak Profit": round(peak_profit, 2),
            "Retention %": round(retention, 2)
        }
    except (IndexError, KeyError, TypeError):
        return "Error calculating metrics."