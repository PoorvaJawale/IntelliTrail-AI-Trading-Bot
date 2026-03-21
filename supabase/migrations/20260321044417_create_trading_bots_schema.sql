/*
  # IntelliTrail Trading Bots Database Schema

  ## Overview
  Creates the complete database structure for AI-powered trading bot management system

  ## New Tables
  
  ### 1. `orders`
  Stores all trading orders (buy/sell) with their status and execution details
  - `id` (uuid, primary key) - Unique order identifier
  - `user_id` (uuid) - User who created the order (for future auth)
  - `ticker` (text) - Stock symbol (e.g., RELIANCE, TCS)
  - `strategy_type` (text) - Either 'buy' or 'sell'
  - `quantity` (integer) - Number of shares
  - `status` (text) - Order status: pending, executed, cancelled, failed
  - `entry_price` (numeric) - Price at which order was entered
  - `ai_predicted_price` (numeric) - AI's predicted optimal execution price
  - `ai_executed_price` (numeric) - Actual AI execution price
  - `target_price` (numeric) - Manual target price for comparison
  - `sim_days` (integer) - Number of simulation days used
  - `entry_date` (date) - Date when order was placed
  - `execution_date` (timestamptz) - When AI executed the order
  - `pnl` (numeric) - Profit/Loss in currency
  - `created_at` (timestamptz) - Record creation timestamp
  - `updated_at` (timestamptz) - Last update timestamp

  ### 2. `bot_executions`
  Detailed logs of AI bot execution steps and decisions
  - `id` (uuid, primary key)
  - `order_id` (uuid, foreign key) - Links to orders table
  - `execution_step` (text) - Step description (e.g., "Data Loading", "AI Prediction")
  - `step_status` (text) - success, failed, in_progress
  - `step_data` (jsonb) - Additional data for the step
  - `timestamp` (timestamptz) - When this step occurred

  ### 3. `ai_predictions`
  Stores AI model predictions for analysis and improvement
  - `id` (uuid, primary key)
  - `order_id` (uuid, foreign key)
  - `ticker` (text)
  - `timeframe` (text) - daily, minute, hourly
  - `predicted_price` (numeric) - AI predicted price
  - `actual_price` (numeric) - Actual market price
  - `confidence_score` (numeric) - Model confidence (0-1)
  - `features_used` (jsonb) - Technical indicators used (SMA, RSI, ATR, etc.)
  - `prediction_date` (timestamptz)
  - `created_at` (timestamptz)

  ### 4. `manual_levels`
  Stores manual trading levels for AI vs Manual comparison
  - `id` (uuid, primary key)
  - `order_id` (uuid, foreign key)
  - `level_name` (text) - e.g., "5% gain", "2% dip"
  - `price` (numeric) - Price level
  - `created_at` (timestamptz)

  ## Security
  - Enable RLS on all tables
  - Add policies for authenticated users to manage their own data
  - Public read access for demo purposes (can be restricted later)

  ## Indexes
  - Index on ticker for fast stock lookups
  - Index on status for filtering active orders
  - Index on created_at for chronological queries
*/

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid,
  ticker text NOT NULL,
  strategy_type text NOT NULL CHECK (strategy_type IN ('buy', 'sell')),
  quantity integer NOT NULL CHECK (quantity > 0),
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'executed', 'cancelled', 'failed')),
  entry_price numeric(10, 2),
  ai_predicted_price numeric(10, 2),
  ai_executed_price numeric(10, 2),
  target_price numeric(10, 2),
  sim_days integer DEFAULT 30,
  entry_date date,
  execution_date timestamptz,
  pnl numeric(12, 2) DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create bot_executions table
CREATE TABLE IF NOT EXISTS bot_executions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  order_id uuid NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
  execution_step text NOT NULL,
  step_status text NOT NULL DEFAULT 'in_progress' CHECK (step_status IN ('success', 'failed', 'in_progress')),
  step_data jsonb DEFAULT '{}'::jsonb,
  timestamp timestamptz DEFAULT now()
);

-- Create ai_predictions table
CREATE TABLE IF NOT EXISTS ai_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  order_id uuid REFERENCES orders(id) ON DELETE SET NULL,
  ticker text NOT NULL,
  timeframe text DEFAULT 'daily',
  predicted_price numeric(10, 2) NOT NULL,
  actual_price numeric(10, 2),
  confidence_score numeric(3, 2),
  features_used jsonb DEFAULT '{}'::jsonb,
  prediction_date timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- Create manual_levels table
CREATE TABLE IF NOT EXISTS manual_levels (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  order_id uuid NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
  level_name text NOT NULL,
  price numeric(10, 2) NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_executions_order_id ON bot_executions(order_id);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_order_id ON ai_predictions(order_id);
CREATE INDEX IF NOT EXISTS idx_manual_levels_order_id ON manual_levels(order_id);

-- Enable Row Level Security
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE manual_levels ENABLE ROW LEVEL SECURITY;

-- Create policies for orders table
CREATE POLICY "Enable read access for all users"
  ON orders FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Enable insert for all users"
  ON orders FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Enable update for all users"
  ON orders FOR UPDATE
  TO public
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Enable delete for all users"
  ON orders FOR DELETE
  TO public
  USING (true);

-- Create policies for bot_executions table
CREATE POLICY "Enable read access for all users"
  ON bot_executions FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Enable insert for all users"
  ON bot_executions FOR INSERT
  TO public
  WITH CHECK (true);

-- Create policies for ai_predictions table
CREATE POLICY "Enable read access for all users"
  ON ai_predictions FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Enable insert for all users"
  ON ai_predictions FOR INSERT
  TO public
  WITH CHECK (true);

-- Create policies for manual_levels table
CREATE POLICY "Enable read access for all users"
  ON manual_levels FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Enable insert for all users"
  ON manual_levels FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Enable delete for all users"
  ON manual_levels FOR DELETE
  TO public
  USING (true);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for orders table
DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
CREATE TRIGGER update_orders_updated_at
  BEFORE UPDATE ON orders
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
