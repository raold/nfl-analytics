-- Prediction Markets Trading Schema
-- Created: February 13, 2026
-- Purpose: Support Kalshi prediction market trading system
--
-- Three new schemas:
--   trading  — Orders, positions, fills, P&L tracking
--   markets  — Contract metadata, events, price history (TimescaleDB)
--   risk     — Limits, breaches, kill switch events
--
-- All monetary values stored in cents (integer) to avoid floating-point rounding.
-- Kalshi contract prices: 1-99 cents (1¢ = 1% implied probability).

-- ============================================================
-- MARKETS SCHEMA — Contract and price data
-- ============================================================

CREATE SCHEMA IF NOT EXISTS markets;

-- Sports events (games, tournaments, etc.)
CREATE TABLE IF NOT EXISTS markets.events (
    event_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL CHECK (sport IN ('nfl', 'nba', 'mlb', 'nhl')),
    game_id TEXT,  -- FK to public.games for NFL
    home_team TEXT,
    away_team TEXT,
    title TEXT NOT NULL,
    scheduled_at TIMESTAMPTZ,
    settled_at TIMESTAMPTZ,
    is_settled BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_sport ON markets.events (sport);
CREATE INDEX IF NOT EXISTS idx_events_game_id ON markets.events (game_id);
CREATE INDEX IF NOT EXISTS idx_events_scheduled ON markets.events (scheduled_at);

-- Individual contracts within events
CREATE TABLE IF NOT EXISTS markets.contracts (
    contract_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES markets.events(event_id),
    title TEXT NOT NULL,
    contract_type TEXT NOT NULL CHECK (
        contract_type IN ('spread', 'moneyline', 'total', 'player_prop', 'futures', 'custom')
    ),
    sport TEXT NOT NULL,
    strike NUMERIC,  -- For spreads/totals (e.g., -3.5, 45.5)
    expiry TIMESTAMPTZ,
    last_price_cents INT CHECK (last_price_cents BETWEEN 0 AND 100),
    volume INT NOT NULL DEFAULT 0,
    open_interest INT NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    outcome_yes BOOLEAN,  -- NULL until settled, TRUE/FALSE after
    kalshi_ticker TEXT,  -- e.g., 'KXNFL-26-AFC-KC'
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_contracts_event ON markets.contracts (event_id);
CREATE INDEX IF NOT EXISTS idx_contracts_sport ON markets.contracts (sport);
CREATE INDEX IF NOT EXISTS idx_contracts_active ON markets.contracts (is_active) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_contracts_ticker ON markets.contracts (kalshi_ticker);

-- Price history — TimescaleDB hypertable for time-series queries
CREATE TABLE IF NOT EXISTS markets.price_history (
    recorded_at TIMESTAMPTZ NOT NULL,
    contract_id TEXT NOT NULL,
    yes_bid_cents INT,
    yes_ask_cents INT,
    last_price_cents INT,
    volume INT DEFAULT 0,
    open_interest INT DEFAULT 0,
    source TEXT DEFAULT 'kalshi'  -- 'kalshi', 'websocket', 'paper'
);

-- Convert to TimescaleDB hypertable (chunks by time for fast range queries)
-- Note: This will fail gracefully if TimescaleDB is not installed
DO $$
BEGIN
    PERFORM create_hypertable(
        'markets.price_history',
        'recorded_at',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
    );
EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'TimescaleDB not available, using regular table for price_history';
END;
$$;

CREATE INDEX IF NOT EXISTS idx_price_history_contract
    ON markets.price_history (contract_id, recorded_at DESC);


-- ============================================================
-- TRADING SCHEMA — Orders, positions, fills, P&L
-- ============================================================

CREATE SCHEMA IF NOT EXISTS trading;

-- All orders (submitted, filled, cancelled, etc.)
CREATE TABLE IF NOT EXISTS trading.orders (
    order_id TEXT PRIMARY KEY,
    contract_id TEXT NOT NULL,
    exchange_order_id TEXT,  -- Exchange-assigned ID (NULL for paper)
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    order_type TEXT NOT NULL DEFAULT 'limit' CHECK (order_type IN ('limit')),
    price_cents INT NOT NULL CHECK (price_cents BETWEEN 1 AND 99),
    quantity INT NOT NULL CHECK (quantity >= 1),
    filled_quantity INT NOT NULL DEFAULT 0,
    avg_fill_price_cents INT DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'open', 'partial', 'filled', 'cancelled', 'rejected')
    ),
    strategy TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'paper' CHECK (mode IN ('paper', 'live')),
    -- Signal metadata (from strategy)
    edge_cents INT,
    model_prob NUMERIC(5, 4),  -- 0.0000 to 1.0000
    confidence NUMERIC(5, 4),
    meta_prob NUMERIC(5, 4),
    kelly_fraction NUMERIC(6, 4),
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    filled_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_orders_contract ON trading.orders (contract_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading.orders (status) WHERE status IN ('open', 'partial', 'pending');
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON trading.orders (strategy);
CREATE INDEX IF NOT EXISTS idx_orders_created ON trading.orders (created_at DESC);

-- Immutable fill history (one row per execution)
CREATE TABLE IF NOT EXISTS trading.fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES trading.orders(order_id),
    contract_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    price_cents INT NOT NULL CHECK (price_cents BETWEEN 1 AND 99),
    quantity INT NOT NULL CHECK (quantity >= 1),
    fee_cents INT NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'paper',
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fills_order ON trading.fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_contract ON trading.fills (contract_id);
CREATE INDEX IF NOT EXISTS idx_fills_executed ON trading.fills (executed_at DESC);

-- Open positions (one row per contract, updated on fills and settlements)
CREATE TABLE IF NOT EXISTS trading.positions (
    contract_id TEXT PRIMARY KEY,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    quantity INT NOT NULL DEFAULT 0,
    avg_entry_cents INT NOT NULL DEFAULT 0,
    cost_basis_cents INT NOT NULL DEFAULT 0,
    realized_pnl_cents INT NOT NULL DEFAULT 0,
    market_id TEXT,
    sport TEXT DEFAULT 'nfl',
    strategy TEXT DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'paper',
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_sport ON trading.positions (sport);
CREATE INDEX IF NOT EXISTS idx_positions_active ON trading.positions (quantity) WHERE quantity > 0;

-- Daily P&L snapshots for equity curve tracking
CREATE TABLE IF NOT EXISTS trading.pnl_snapshots (
    snapshot_date DATE NOT NULL,
    mode TEXT NOT NULL DEFAULT 'paper',
    -- Equity
    total_equity_cents INT NOT NULL,
    cash_balance_cents INT NOT NULL,
    unrealized_pnl_cents INT NOT NULL DEFAULT 0,
    realized_pnl_cents INT NOT NULL DEFAULT 0,
    -- Day metrics
    daily_pnl_cents INT NOT NULL DEFAULT 0,
    daily_trades INT NOT NULL DEFAULT 0,
    daily_volume_cents INT NOT NULL DEFAULT 0,
    -- Cumulative stats
    total_trades INT NOT NULL DEFAULT 0,
    win_count INT NOT NULL DEFAULT 0,
    loss_count INT NOT NULL DEFAULT 0,
    -- Risk metrics
    max_drawdown_pct NUMERIC(6, 4) DEFAULT 0,
    sharpe_30d NUMERIC(6, 4),
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, mode)
);


-- ============================================================
-- RISK SCHEMA — Limits, breaches, kill switch
-- ============================================================

CREATE SCHEMA IF NOT EXISTS risk;

-- Configurable risk limits
CREATE TABLE IF NOT EXISTS risk.limits (
    limit_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    limit_type TEXT NOT NULL CHECK (
        limit_type IN (
            'max_position_pct',     -- Max % of bankroll per position
            'max_exposure_pct',     -- Max % total exposure
            'max_daily_loss_pct',   -- Daily stop loss %
            'max_drawdown_pct',     -- Peak-to-trough drawdown %
            'max_correlation_pct',  -- Max in correlated group
            'max_orders_per_sec',   -- Rate limit
            'max_open_orders'       -- Total open orders
        )
    ),
    sport TEXT,  -- NULL = applies to all sports
    value NUMERIC NOT NULL,
    unit TEXT NOT NULL DEFAULT 'pct',  -- 'pct', 'count', 'cents'
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed default limits
INSERT INTO risk.limits (name, limit_type, value, unit, description) VALUES
    ('max_single_position', 'max_position_pct', 5.0, 'pct', 'Max 5% of bankroll per contract'),
    ('max_total_exposure', 'max_exposure_pct', 25.0, 'pct', 'Max 25% of bankroll in open positions'),
    ('daily_stop_loss', 'max_daily_loss_pct', 10.0, 'pct', 'Halt trading at 10% daily loss'),
    ('max_drawdown', 'max_drawdown_pct', 20.0, 'pct', 'Kill switch at 20% peak drawdown'),
    ('correlation_limit', 'max_correlation_pct', 10.0, 'pct', 'Max 10% in correlated contracts'),
    ('rate_limit', 'max_orders_per_sec', 20.0, 'count', 'Kalshi rate limit: 20 orders/sec'),
    ('open_order_limit', 'max_open_orders', 200000.0, 'count', 'Kalshi max 200K open orders')
ON CONFLICT (name) DO NOTHING;

-- Breach log (when an order or action violates a limit)
CREATE TABLE IF NOT EXISTS risk.breaches (
    breach_id SERIAL PRIMARY KEY,
    limit_id INT NOT NULL REFERENCES risk.limits(limit_id),
    order_id TEXT,  -- NULL if not order-specific
    contract_id TEXT,
    attempted_value NUMERIC NOT NULL,
    limit_value NUMERIC NOT NULL,
    action_taken TEXT NOT NULL CHECK (
        action_taken IN ('blocked', 'reduced', 'warned', 'kill_switch')
    ),
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_breaches_limit ON risk.breaches (limit_id);
CREATE INDEX IF NOT EXISTS idx_breaches_created ON risk.breaches (created_at DESC);

-- Kill switch events (critical — when the system emergency-halts)
CREATE TABLE IF NOT EXISTS risk.kill_switch_log (
    event_id SERIAL PRIMARY KEY,
    trigger_reason TEXT NOT NULL,
    trigger_limit_id INT REFERENCES risk.limits(limit_id),
    loss_at_trigger_cents INT,
    drawdown_at_trigger_pct NUMERIC(6, 4),
    positions_closed INT NOT NULL DEFAULT 0,
    orders_cancelled INT NOT NULL DEFAULT 0,
    action_taken TEXT NOT NULL,
    auto_resume_at TIMESTAMPTZ,  -- When to auto-resume (NULL = manual)
    resolved_at TIMESTAMPTZ,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- HELPER VIEWS
-- ============================================================

-- Active positions with current P&L
CREATE OR REPLACE VIEW trading.v_active_positions AS
SELECT
    p.contract_id,
    p.side,
    p.quantity,
    p.avg_entry_cents,
    p.cost_basis_cents,
    p.realized_pnl_cents,
    p.strategy,
    p.mode,
    c.title AS contract_title,
    c.contract_type,
    c.sport,
    c.last_price_cents,
    c.expiry,
    CASE
        WHEN p.side = 'yes' THEN (c.last_price_cents - p.avg_entry_cents) * p.quantity
        ELSE (p.avg_entry_cents - c.last_price_cents) * p.quantity
    END AS unrealized_pnl_cents
FROM trading.positions p
LEFT JOIN markets.contracts c ON c.contract_id = p.contract_id
WHERE p.quantity > 0;

-- Today's trading activity
CREATE OR REPLACE VIEW trading.v_today_activity AS
SELECT
    o.strategy,
    o.mode,
    COUNT(*) AS total_orders,
    COUNT(*) FILTER (WHERE o.status = 'filled') AS filled_orders,
    COUNT(*) FILTER (WHERE o.status IN ('open', 'partial')) AS open_orders,
    SUM(o.filled_quantity) AS total_filled_qty,
    AVG(o.edge_cents) AS avg_edge_cents,
    AVG(o.model_prob) AS avg_model_prob
FROM trading.orders o
WHERE o.created_at >= CURRENT_DATE
GROUP BY o.strategy, o.mode;

-- Equity curve (for Grafana dashboards)
CREATE OR REPLACE VIEW trading.v_equity_curve AS
SELECT
    snapshot_date,
    mode,
    total_equity_cents / 100.0 AS total_equity_dollars,
    cash_balance_cents / 100.0 AS cash_dollars,
    unrealized_pnl_cents / 100.0 AS unrealized_pnl_dollars,
    realized_pnl_cents / 100.0 AS realized_pnl_dollars,
    daily_pnl_cents / 100.0 AS daily_pnl_dollars,
    daily_trades,
    max_drawdown_pct,
    sharpe_30d
FROM trading.pnl_snapshots
ORDER BY snapshot_date;
