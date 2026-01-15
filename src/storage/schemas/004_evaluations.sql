-- PyFlare ClickHouse Schema: Evaluations Table
-- Migration: 004_evaluations.sql
-- Description: Stores evaluation results from various evaluators

-- Evaluations table
CREATE TABLE IF NOT EXISTS pyflare.evaluations
(
    -- Evaluation identity
    evaluation_id UUID DEFAULT generateUUIDv4(),

    -- Link to trace
    trace_id String,
    span_id String,

    -- Evaluation type
    evaluator_type LowCardinality(String),  -- 'hallucination', 'rag_quality', 'toxicity', 'custom'

    -- Results
    score Float64,
    verdict LowCardinality(String),  -- 'pass', 'warn', 'fail', 'error'
    confidence Float64 DEFAULT 1.0,

    -- Detailed scores (JSON object)
    sub_scores String DEFAULT '{}',

    -- Human-readable explanation
    explanation String DEFAULT '',

    -- Context
    model_id LowCardinality(String) DEFAULT '',
    service_name LowCardinality(String) DEFAULT '',
    user_id String DEFAULT '',

    -- Input/output that was evaluated (for debugging)
    input_sample String DEFAULT '',
    output_sample String DEFAULT '',

    -- For RAG evaluations
    retrieved_contexts String DEFAULT '[]',  -- JSON array

    -- Evaluation metadata
    evaluator_model LowCardinality(String) DEFAULT '',  -- Model used for LLM-as-judge
    evaluator_version String DEFAULT '',

    -- Timing
    evaluated_at DateTime64(3),
    evaluation_latency_ms Float64 DEFAULT 0.0,

    -- Partitioning
    event_date Date DEFAULT toDate(evaluated_at),

    -- Timestamps
    created_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (evaluator_type, model_id, evaluated_at, trace_id)
TTL event_date + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Indices
ALTER TABLE pyflare.evaluations
    ADD INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 4,
    ADD INDEX idx_verdict verdict TYPE set(4) GRANULARITY 4;

-- Materialized view for evaluation summaries by model
CREATE MATERIALIZED VIEW IF NOT EXISTS pyflare.mv_evaluation_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (evaluator_type, model_id, event_date)
AS SELECT
    event_date,
    evaluator_type,
    model_id,
    service_name,
    count() AS total_count,
    countIf(verdict = 'pass') AS pass_count,
    countIf(verdict = 'warn') AS warn_count,
    countIf(verdict = 'fail') AS fail_count,
    countIf(verdict = 'error') AS error_count,
    avg(score) AS avg_score,
    avg(evaluation_latency_ms) AS avg_latency_ms
FROM pyflare.evaluations
GROUP BY event_date, evaluator_type, model_id, service_name;

-- RCA (Root Cause Analysis) reports table
CREATE TABLE IF NOT EXISTS pyflare.rca_reports
(
    -- Report identity
    report_id UUID DEFAULT generateUUIDv4(),

    -- Scope
    model_id LowCardinality(String),
    service_name LowCardinality(String) DEFAULT '',
    environment LowCardinality(String) DEFAULT '',

    -- Analysis time window
    analysis_start DateTime64(3),
    analysis_end DateTime64(3),

    -- Generated at
    generated_at DateTime64(3),

    -- Overall metrics
    total_records UInt64,
    failed_records UInt64,
    overall_failure_rate Float64,

    -- Analysis results (JSON)
    problematic_slices String DEFAULT '[]',  -- Array of slice objects
    patterns String DEFAULT '[]',             -- Array of pattern objects
    suspected_causes String DEFAULT '[]',     -- Array of root cause objects

    -- Recommendations
    recommendations String DEFAULT '[]',  -- JSON array of strings

    -- Report status
    status LowCardinality(String) DEFAULT 'generated',  -- 'generated', 'reviewed', 'actioned'

    -- Partitioning
    event_date Date DEFAULT toDate(generated_at),

    -- Timestamps
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(event_date)
ORDER BY (model_id, report_id)
TTL event_date + INTERVAL 365 DAY
SETTINGS index_granularity = 8192;

-- Failure records for RCA analysis
CREATE TABLE IF NOT EXISTS pyflare.failure_records
(
    -- Identity
    failure_id UUID DEFAULT generateUUIDv4(),
    trace_id String,
    span_id String,

    -- Failure details
    model_id LowCardinality(String),
    service_name LowCardinality(String) DEFAULT '',
    failure_type LowCardinality(String),  -- 'error', 'timeout', 'hallucination', 'toxic', etc.

    -- Context
    input_preview String DEFAULT '',
    output_preview String DEFAULT '',
    error_message String DEFAULT '',

    -- Attributes for slicing
    user_id String DEFAULT '',
    feature_id String DEFAULT '',
    input_length UInt32 DEFAULT 0,
    output_length UInt32 DEFAULT 0,

    -- Timing
    occurred_at DateTime64(3),

    -- Embedding for clustering (optional)
    embedding Array(Float32) DEFAULT [],

    -- Cluster assignment (from RCA)
    cluster_id String DEFAULT '',

    -- Partitioning
    event_date Date DEFAULT toDate(occurred_at)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (model_id, failure_type, occurred_at)
TTL event_date + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- View for model health summary
CREATE VIEW IF NOT EXISTS pyflare.model_health_summary AS
SELECT
    model_id,
    service_name,
    event_date,
    countIf(status_code = 'ok') AS success_count,
    countIf(status_code = 'error') AS error_count,
    count() AS total_count,
    error_count / total_count AS error_rate,
    avg(duration_ns) / 1000000.0 AS avg_latency_ms,
    quantile(0.95)(duration_ns) / 1000000.0 AS p95_latency_ms,
    avgIf(hallucination_score, hallucination_score >= 0) AS avg_hallucination_score,
    avgIf(toxicity_score, toxicity_score >= 0) AS avg_toxicity_score,
    avgIf(embedding_drift_score, embedding_drift_score >= 0) AS avg_drift_score,
    sum(cost_micros) AS total_cost_micros
FROM pyflare.traces
WHERE event_date >= today() - 7
GROUP BY model_id, service_name, event_date
ORDER BY event_date DESC, model_id;
