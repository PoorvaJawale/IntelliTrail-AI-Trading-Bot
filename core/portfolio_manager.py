from core.engine import run_simulation_with_ai, run_buy_simulation

class PortfolioManager:
    def __init__(self):
        self.active_bots = {}

    def add_bot(self, ticker, strategy_type, target_price):
        """
        Adds a new bot to the portfolio.
        strategy_type: 'SELL' or 'BUY'
        """
        self.active_stocks[ticker] = {
            'strategy': strategy_type,
            'target': target_price,
            'status': 'Running'
        }

    def run_all(self, data_dict, model):
        """
        Runs all bots simultaneously against their respective data.
        """
        results = {}
        for ticker, config in self.active_stocks.items():
            df = data_dict.get(ticker)
            if config['strategy'] == 'SELL':
                results[ticker] = run_simulation_with_ai(df, config['target'], model)
            else:
                results[ticker] = run_buy_simulation(df, config['target'])
        return results