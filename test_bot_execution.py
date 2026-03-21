import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import joblib

load_dotenv()

from core.bot_orchestrator import BotOrchestrator

def test_bot_execution():
    """Test the complete bot execution pipeline"""

    print("Initializing Supabase connection...")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        print("ERROR: Supabase credentials not found in .env file")
        return

    supabase = create_client(url, key)
    print("Supabase connected successfully")

    print("\nLoading AI model...")
    model_path = 'models/trained_models/nifty_model.pkl'
    model = joblib.load(model_path)
    print("Model loaded successfully")

    print("\nInitializing Bot Orchestrator...")
    orchestrator = BotOrchestrator(model, supabase)
    print("Bot Orchestrator ready")

    print("\n" + "="*60)
    print("TEST 1: BUY ORDER EXECUTION")
    print("="*60)

    buy_order = {
        'ticker': 'RELIANCE',
        'strategy_type': 'buy',
        'quantity': 50,
        'sim_days': 30
    }

    print(f"\nExecuting BUY order for {buy_order['ticker']}...")
    buy_result = orchestrator.execute_bot(buy_order)

    if buy_result['success']:
        print("\nBUY Order executed successfully!")
        print(f"  Order ID: {buy_result['order_id']}")
        print(f"  Ticker: {buy_result['ticker']}")
        print(f"  Executed Price: ₹{buy_result['executed_price']:,.2f}")
        print(f"  Execution Date: {buy_result['execution_date']}")
        print(f"  Confidence: {buy_result['confidence']:.1%}")
        print(f"  P&L: ₹{buy_result['pnl']:,.2f}")
        print("\n  Manual Comparison Levels:")
        for level, price in buy_result['manual_levels'].items():
            print(f"    {level}: ₹{price:,.2f}")
    else:
        print(f"\nBUY Order failed: {buy_result.get('error')}")

    print("\n" + "="*60)
    print("TEST 2: SELL ORDER EXECUTION")
    print("="*60)

    sell_order = {
        'ticker': 'TCS',
        'strategy_type': 'sell',
        'quantity': 20,
        'sim_days': 30
    }

    print(f"\nExecuting SELL order for {sell_order['ticker']}...")
    sell_result = orchestrator.execute_bot(sell_order)

    if sell_result['success']:
        print("\nSELL Order executed successfully!")
        print(f"  Order ID: {sell_result['order_id']}")
        print(f"  Ticker: {sell_result['ticker']}")
        print(f"  Executed Price: ₹{sell_result['executed_price']:,.2f}")
        print(f"  Execution Date: {sell_result['execution_date']}")
        print(f"  Confidence: {sell_result['confidence']:.1%}")
        print(f"  P&L: ₹{sell_result['pnl']:,.2f}")
        print("\n  Manual Comparison Levels:")
        for level, price in sell_result['manual_levels'].items():
            print(f"    {level}: ₹{price:,.2f}")
    else:
        print(f"\nSELL Order failed: {sell_result.get('error')}")

    print("\n" + "="*60)
    print("TEST 3: FETCH ALL ORDERS FROM DATABASE")
    print("="*60)

    orders = supabase.table('orders').select('*').execute()
    print(f"\nTotal orders in database: {len(orders.data)}")

    for order in orders.data:
        print(f"\n  {order['ticker']} | {order['strategy_type'].upper()} | {order['status'].upper()}")
        print(f"    Quantity: {order['quantity']}")
        print(f"    Executed Price: ₹{order.get('ai_executed_price', 0):,.2f}")
        print(f"    P&L: ₹{order.get('pnl', 0):,.2f}")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_bot_execution()
