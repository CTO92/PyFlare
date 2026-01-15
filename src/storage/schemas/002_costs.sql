-- PyFlare ClickHouse Schema: Costs Table
-- Migration: 002_costs.sql
-- Description: Aggregated cost data for analytics and budgeting

-- Costs table: pre-aggregated cost data for efficient analytics
CREATE TABLE IF NOT EXISTS pyflare.costs
(
    -- Time bucket
    timestamp DateTime,
    bucket_period LowCardinality(String),  -- 'minute', 'hour', 'day'

    -- Dimensions
    model_id LowCardinality(String),
    model_provider LowCardinality(String),
    service_name LowCardinality(String),
    user_id String DEFAULT '',
    feature_id String DEFAULT '',
    environment LowCardinality(String) DEFAULT 'development',

    -- Token metrics
    request_count UInt64 DEFAULT 0,
    input_tokens UInt64 DEFAULT 0,
    output_tokens UInt64 DEFAULT 0,
    total_tokens UInt64 DEFAULT 0,
    cached_tokens UInt64 DEFAULT 0,

    -- Cost metrics (in micro-dollars)
    input_cost_micros Int64 DEFAULT 0,
    output_cost_micros Int64 DEFAULT 0,
    total_cost_micros Int64 DEFAULT 0,

    -- Latency metrics (milliseconds)
    avg_latency_ms Float64 DEFAULT 0.0,
    p50_latency_ms Float64 DEFAULT 0.0,
    p95_latency_ms Float64 DEFAULT 0.0,
    p99_latency_ms Float64 DEFAULT 0.0,
    max_latency_ms Float64 DEFAULT 0.0,

    -- Error metrics
    error_count UInt64 DEFAULT 0,
    error_rate Float64 DEFAULT 0.0,

    -- Partitioning
    event_date Date DEFAULT toDate(timestamp)
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (bucket_period, model_id, service_name, timestamp)
TTL event_date + INTERVAL 365 DAY
SETTINGS index_granularity = 8192;

-- Model pricing table: stores pricing information for cost calculation
CREATE TABLE IF NOT EXISTS pyflare.model_pricing
(
    model_id LowCardinality(String),
    model_provider LowCardinality(String),

    -- Pricing per million tokens (in micro-dollars)
    input_price_per_million Int64,
    output_price_per_million Int64,
    cached_input_price_per_million Int64 DEFAULT 0,

    -- Effective date range
    effective_from DateTime,
    effective_until Nullable(DateTime),

    -- Metadata
    currency LowCardinality(String) DEFAULT 'USD',
    notes String DEFAULT '',
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (model_id, effective_from)
SETTINGS index_granularity = 8192;

-- Insert default pricing data (as of 2024)
INSERT INTO pyflare.model_pricing (model_id, model_provider, input_price_per_million, output_price_per_million, effective_from) VALUES
    -- OpenAI models
    ('gpt-4o', 'openai', 2500000, 10000000, '2024-01-01 00:00:00'),
    ('gpt-4o-mini', 'openai', 150000, 600000, '2024-01-01 00:00:00'),
    ('gpt-4-turbo', 'openai', 10000000, 30000000, '2024-01-01 00:00:00'),
    ('gpt-4', 'openai', 30000000, 60000000, '2024-01-01 00:00:00'),
    ('gpt-3.5-turbo', 'openai', 500000, 1500000, '2024-01-01 00:00:00'),
    ('text-embedding-3-small', 'openai', 20000, 0, '2024-01-01 00:00:00'),
    ('text-embedding-3-large', 'openai', 130000, 0, '2024-01-01 00:00:00'),

    -- Anthropic models
    ('claude-3-opus', 'anthropic', 15000000, 75000000, '2024-01-01 00:00:00'),
    ('claude-3-sonnet', 'anthropic', 3000000, 15000000, '2024-01-01 00:00:00'),
    ('claude-3-haiku', 'anthropic', 250000, 1250000, '2024-01-01 00:00:00'),
    ('claude-3-5-sonnet', 'anthropic', 3000000, 15000000, '2024-01-01 00:00:00'),

    -- Google models
    ('gemini-1.5-pro', 'google', 3500000, 10500000, '2024-01-01 00:00:00'),
    ('gemini-1.5-flash', 'google', 75000, 300000, '2024-01-01 00:00:00'),

    -- Mistral models
    ('mistral-large', 'mistral', 4000000, 12000000, '2024-01-01 00:00:00'),
    ('mistral-medium', 'mistral', 2700000, 8100000, '2024-01-01 00:00:00'),
    ('mistral-small', 'mistral', 1000000, 3000000, '2024-01-01 00:00:00');

-- Budget configuration table
CREATE TABLE IF NOT EXISTS pyflare.budgets
(
    budget_id String,
    name String,

    -- Budget scope
    dimension LowCardinality(String),  -- 'global', 'model', 'user', 'team', 'feature', 'service'
    dimension_value String DEFAULT '',

    -- Limits (in micro-dollars)
    daily_limit_micros Int64 DEFAULT 0,
    weekly_limit_micros Int64 DEFAULT 0,
    monthly_limit_micros Int64 DEFAULT 0,

    -- Alert thresholds (percentage of limit)
    warning_threshold Float64 DEFAULT 0.8,
    critical_threshold Float64 DEFAULT 0.95,

    -- Alert channels (JSON array)
    alert_channels String DEFAULT '[]',

    -- Status
    is_active UInt8 DEFAULT 1,

    -- Timestamps
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (budget_id)
SETTINGS index_granularity = 8192;

-- Budget spend tracking table
CREATE TABLE IF NOT EXISTS pyflare.budget_spend
(
    budget_id String,
    period_start DateTime,
    period_type LowCardinality(String),  -- 'daily', 'weekly', 'monthly'

    -- Current spend
    current_spend_micros Int64 DEFAULT 0,

    -- Tracking
    last_updated DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (budget_id, period_type, period_start)
SETTINGS index_granularity = 8192;

-- Materialized view for daily cost summary
CREATE MATERIALIZED VIEW IF NOT EXISTS pyflare.mv_daily_costs
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (model_id, service_name, event_date)
AS SELECT
    event_date,
    model_id,
    model_provider,
    service_name,
    user_id,
    count() AS request_count,
    sum(input_tokens) AS total_input_tokens,
    sum(output_tokens) AS total_output_tokens,
    sum(cost_micros) AS total_cost_micros
FROM pyflare.traces
GROUP BY event_date, model_id, model_provider, service_name, user_id;
