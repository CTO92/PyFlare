-- PyFlare ClickHouse Schema: Traces Table
-- Migration: 001_traces.sql
-- Description: Primary table for storing trace/span data

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS pyflare;

-- Traces table: stores all span data from the collector
CREATE TABLE IF NOT EXISTS pyflare.traces
(
    -- Identity
    trace_id String,
    span_id String,
    parent_span_id String DEFAULT '',

    -- Timing (nanosecond precision)
    start_time DateTime64(9),
    end_time DateTime64(9),
    duration_ns UInt64 MATERIALIZED toUInt64((end_time - start_time) * 1000000000),

    -- Service information
    service_name LowCardinality(String),
    span_name String,
    span_kind LowCardinality(String) DEFAULT 'internal',  -- internal, server, client, producer, consumer

    -- Status
    status_code LowCardinality(String) DEFAULT 'unset',  -- unset, ok, error
    status_message String DEFAULT '',

    -- ML-specific attributes
    model_id LowCardinality(String) DEFAULT '',
    model_provider LowCardinality(String) DEFAULT '',
    model_version String DEFAULT '',
    inference_type LowCardinality(String) DEFAULT '',  -- llm, embedding, classification, etc.

    -- Token usage
    input_tokens UInt32 DEFAULT 0,
    output_tokens UInt32 DEFAULT 0,
    total_tokens UInt32 DEFAULT 0,

    -- Cost (in micro-dollars, 1/1,000,000 USD)
    cost_micros Int64 DEFAULT 0,

    -- Attribution
    user_id String DEFAULT '',
    feature_id String DEFAULT '',
    session_id String DEFAULT '',
    environment LowCardinality(String) DEFAULT 'development',

    -- Input/Output previews (truncated for storage efficiency)
    input_preview String DEFAULT '',
    output_preview String DEFAULT '',

    -- Evaluation scores (populated by evaluators)
    hallucination_score Float32 DEFAULT -1.0,  -- -1 = not evaluated
    toxicity_score Float32 DEFAULT -1.0,
    rag_relevance_score Float32 DEFAULT -1.0,
    rag_groundedness_score Float32 DEFAULT -1.0,

    -- Drift scores (populated by drift detectors)
    embedding_drift_score Float32 DEFAULT -1.0,
    feature_drift_score Float32 DEFAULT -1.0,

    -- Resource information
    host_name String DEFAULT '',
    host_ip String DEFAULT '',

    -- Flexible attributes (for custom user attributes)
    attributes Map(String, String) DEFAULT map(),

    -- Partitioning column
    event_date Date DEFAULT toDate(start_time),

    -- Ingestion timestamp
    inserted_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (service_name, model_id, start_time, trace_id)
TTL event_date + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Indices for common query patterns
ALTER TABLE pyflare.traces
    ADD INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 4,
    ADD INDEX idx_user_id user_id TYPE bloom_filter(0.01) GRANULARITY 4,
    ADD INDEX idx_status_code status_code TYPE set(3) GRANULARITY 4,
    ADD INDEX idx_model_provider model_provider TYPE set(20) GRANULARITY 4;

-- View for quick trace lookup
CREATE VIEW IF NOT EXISTS pyflare.traces_by_trace_id AS
SELECT *
FROM pyflare.traces
ORDER BY trace_id, start_time;

-- Materialized view for hourly aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS pyflare.mv_traces_hourly
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (service_name, model_id, hour)
AS SELECT
    toStartOfHour(start_time) AS hour,
    service_name,
    model_id,
    model_provider,
    count() AS request_count,
    sum(input_tokens) AS total_input_tokens,
    sum(output_tokens) AS total_output_tokens,
    sum(cost_micros) AS total_cost_micros,
    avg(duration_ns) / 1000000.0 AS avg_latency_ms,
    countIf(status_code = 'error') AS error_count
FROM pyflare.traces
GROUP BY hour, service_name, model_id, model_provider;
