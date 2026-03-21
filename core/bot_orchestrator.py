import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data_loader import load_stock_data
from core.indicators import apply_indicators

class BotOrchestrator:
    """
    Central AI Bot Manager that orchestrates the entire trading pipeline:
    1. Data Loading & Preparation
    2. AI Prediction using trained model
    3. Risk Management (ATR-based)
    4. Optimal Execution Price Calculation
    5. Execution Logging
    """

    def __init__(self, model, supabase_client):
        self.model = model
        self.supabase = supabase_client

    def log_execution_step(self, order_id, step_name, status, data=None):
        """Log each step of bot execution to database"""
        try:
            self.supabase.table('bot_executions').insert({
                'order_id': order_id,
                'execution_step': step_name,
                'step_status': status,
                'step_data': data or {},
                'timestamp': datetime.now().isoformat()
            }).execute()
        except Exception as e:
            print(f"Warning: Failed to log step {step_name}: {e}")

    def calculate_buy_target(self, df, sim_days=30):
        """
        AI-powered BUY logic: Find the absolute minimum price using:
        1. Historical low detection
        2. ATR-based support levels
        3. AI price prediction
        4. Bollinger Band lower bound
        """
        recent_df = df.tail(sim_days).copy()

        # 1. Absolute historical low
        historical_low = recent_df['low'].min()

        # 2. ATR-based support level
        current_atr = recent_df['atr'].iloc[-1]
        current_close = recent_df['close'].iloc[-1]
        atr_support = current_close - (current_atr * 1.5)

        # 3. AI Prediction
        last_row = recent_df.iloc[-1]
        features = pd.DataFrame([[
            last_row['sma_20'],
            last_row['sma_50'],
            last_row['rsi'],
            last_row['atr']
        ]], columns=['sma_20', 'sma_50', 'rsi', 'atr'])

        ai_predicted_price = self.model.predict(features)[0]

        # 4. Bollinger Band lower bound
        bb_period = min(20, len(recent_df))
        bb_std = recent_df['close'].rolling(bb_period).std().iloc[-1]
        bb_lower = current_close - (2 * bb_std)

        # 5. Weighted decision: Take the minimum of all signals
        buy_candidates = [
            historical_low,
            atr_support,
            ai_predicted_price * 0.98,  # AI prediction with 2% safety margin
            bb_lower
        ]

        # Filter out NaN values
        buy_candidates = [x for x in buy_candidates if not np.isnan(x)]

        optimal_buy_price = min(buy_candidates)

        # Find the actual execution point in historical data
        execution_idx = recent_df['low'].idxmin()
        actual_execution_price = recent_df.loc[execution_idx, 'low']

        return {
            'optimal_price': optimal_buy_price,
            'executed_price': actual_execution_price,
            'execution_date': execution_idx,
            'ai_prediction': ai_predicted_price,
            'confidence': self._calculate_confidence(recent_df, actual_execution_price),
            'signals': {
                'historical_low': historical_low,
                'atr_support': atr_support,
                'ai_target': ai_predicted_price * 0.98,
                'bb_lower': bb_lower
            }
        }

    def calculate_sell_target(self, df, entry_price, sim_days=30):
        """
        AI-powered SELL logic: Find the maximum price using:
        1. Dynamic Trailing Stop-Loss (TSLO)
        2. ATR-based resistance
        3. AI price prediction
        4. Risk management
        """
        recent_df = df.tail(sim_days).copy()

        # Initialize TSLO
        initial_atr = recent_df.iloc[0]['atr']
        current_sl = entry_price - (initial_atr * 2.0)
        max_profit = 0
        optimal_exit_price = entry_price
        exit_date = recent_df.index[0]

        tslo_history = []

        for timestamp, row in recent_df.iterrows():
            # AI Prediction for this point
            features = pd.DataFrame([[
                row['sma_20'],
                row['sma_50'],
                row['rsi'],
                row['atr']
            ]], columns=['sma_20', 'sma_50', 'rsi', 'atr'])

            predicted_price = self.model.predict(features)[0]

            # Dynamic multiplier based on AI prediction
            if predicted_price < row['close']:  # Bearish signal
                multiplier = 1.2  # Tighter stop-loss
            else:  # Bullish signal
                multiplier = 2.0  # Looser stop-loss

            # Check if stop-loss hit
            if row['close'] <= current_sl:
                optimal_exit_price = row['close']
                exit_date = timestamp
                break

            # Update trailing stop-loss
            new_sl = row['close'] - (row['atr'] * multiplier)
            if new_sl > current_sl:
                current_sl = new_sl

            tslo_history.append({
                'date': timestamp,
                'price': row['close'],
                'sl': current_sl,
                'multiplier': multiplier
            })

            # Track maximum profit
            profit = row['close'] - entry_price
            if profit > max_profit:
                max_profit = profit
                optimal_exit_price = row['close']
                exit_date = timestamp

        return {
            'optimal_price': optimal_exit_price,
            'executed_price': optimal_exit_price,
            'execution_date': exit_date,
            'max_profit': max_profit,
            'tslo_history': tslo_history,
            'confidence': self._calculate_confidence(recent_df, optimal_exit_price)
        }

    def _calculate_confidence(self, df, target_price):
        """
        Calculate confidence score based on:
        - Distance from current price
        - Volatility (ATR)
        - RSI momentum
        """
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        rsi = df['rsi'].iloc[-1]

        # Price deviation score (closer = higher confidence)
        price_deviation = abs(target_price - current_price) / current_price
        price_score = max(0, 1 - price_deviation * 2)

        # Volatility score (lower volatility = higher confidence)
        volatility_pct = (atr / current_price) * 100
        volatility_score = max(0, 1 - (volatility_pct / 10))

        # RSI momentum score
        if 30 <= rsi <= 70:
            rsi_score = 0.8  # Neutral is good
        else:
            rsi_score = 0.5  # Extreme values = lower confidence

        # Weighted average
        confidence = (price_score * 0.4 + volatility_score * 0.4 + rsi_score * 0.2)
        return round(confidence, 3)

    def execute_bot(self, order_data):
        """
        Main execution pipeline for an AI bot
        Returns: Complete execution results with database integration
        """
        ticker = order_data['ticker']
        strategy_type = order_data['strategy_type']
        quantity = order_data['quantity']
        sim_days = order_data.get('sim_days', 30)

        # Create order in database
        order_record = self.supabase.table('orders').insert({
            'ticker': ticker,
            'strategy_type': strategy_type,
            'quantity': quantity,
            'sim_days': sim_days,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }).execute()

        order_id = order_record.data[0]['id']

        try:
            # Step 1: Load Data
            self.log_execution_step(order_id, 'Data Loading', 'in_progress')

            file_path = f"data/raw/{ticker}_daily.csv"
            df = load_stock_data(file_path)

            if df is None or df.empty:
                raise Exception(f"No data found for {ticker}")

            self.log_execution_step(order_id, 'Data Loading', 'success', {
                'rows_loaded': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}"
            })

            # Step 2: Apply Technical Indicators
            self.log_execution_step(order_id, 'Technical Analysis', 'in_progress')
            df = apply_indicators(df)
            self.log_execution_step(order_id, 'Technical Analysis', 'success', {
                'indicators': ['SMA_20', 'SMA_50', 'RSI', 'ATR']
            })

            # Step 3: AI Execution
            self.log_execution_step(order_id, 'AI Execution', 'in_progress')

            if strategy_type == 'buy':
                result = self.calculate_buy_target(df, sim_days)
                entry_price = result['executed_price']

                # Store manual comparison levels
                manual_levels = {
                    '2% dip': df['close'].iloc[-1] * 0.98,
                    '5% dip': df['close'].iloc[-1] * 0.95,
                    '10% dip': df['close'].iloc[-1] * 0.90
                }

            else:  # sell
                # For sell, we need entry price (assume current price for demo)
                entry_price = df['close'].iloc[-sim_days]
                result = self.calculate_sell_target(df, entry_price, sim_days)

                manual_levels = {
                    '5% gain': entry_price * 1.05,
                    '10% gain': entry_price * 1.10,
                    '2% SL': entry_price * 0.98
                }

            # Calculate P&L
            if strategy_type == 'buy':
                current_price = df['close'].iloc[-1]
                pnl = (current_price - result['executed_price']) * quantity
            else:
                pnl = (result['executed_price'] - entry_price) * quantity

            # Update order with execution results
            self.supabase.table('orders').update({
                'status': 'executed',
                'entry_price': float(entry_price),
                'ai_predicted_price': float(result.get('ai_prediction', result['optimal_price'])),
                'ai_executed_price': float(result['executed_price']),
                'execution_date': result['execution_date'].isoformat(),
                'entry_date': df.index[-sim_days].date().isoformat(),
                'pnl': float(pnl)
            }).eq('id', order_id).execute()

            # Store AI prediction
            self.supabase.table('ai_predictions').insert({
                'order_id': order_id,
                'ticker': ticker,
                'timeframe': 'daily',
                'predicted_price': float(result.get('ai_prediction', result['optimal_price'])),
                'actual_price': float(result['executed_price']),
                'confidence_score': result['confidence'],
                'features_used': {
                    'sma_20': float(df['sma_20'].iloc[-1]),
                    'sma_50': float(df['sma_50'].iloc[-1]),
                    'rsi': float(df['rsi'].iloc[-1]),
                    'atr': float(df['atr'].iloc[-1])
                }
            }).execute()

            # Store manual levels for comparison
            for level_name, price in manual_levels.items():
                self.supabase.table('manual_levels').insert({
                    'order_id': order_id,
                    'level_name': level_name,
                    'price': float(price)
                }).execute()

            self.log_execution_step(order_id, 'AI Execution', 'success', {
                'executed_price': float(result['executed_price']),
                'confidence': result['confidence'],
                'pnl': float(pnl)
            })

            return {
                'success': True,
                'order_id': order_id,
                'ticker': ticker,
                'strategy': strategy_type,
                'executed_price': result['executed_price'],
                'execution_date': result['execution_date'],
                'pnl': pnl,
                'manual_levels': manual_levels,
                'confidence': result['confidence'],
                'details': result
            }

        except Exception as e:
            # Mark as failed
            self.supabase.table('orders').update({
                'status': 'failed'
            }).eq('id', order_id).execute()

            self.log_execution_step(order_id, 'AI Execution', 'failed', {
                'error': str(e)
            })

            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
