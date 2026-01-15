-- PyFlare ClickHouse Schema: Drift Alerts Table
-- Migration: 003_drift_alerts.sql
-- Description: Stores drift detection alerts and history

-- Drift alerts table
CREATE TABLE IF NOT EXISTS pyflare.drift_alerts
(
    -- Alert identity
    alert_id UUID DEFAULT generateUUIDv4(),

    -- When detected
    detected_at DateTime64(3),

    -- Drift details
    drift_type Enum8(
        'feature' = 1,
        'embedding' = 2,
        'concept' = 3,
        'prediction' = 4
    ),
    model_id LowCardinality(String),
    feature_name String DEFAULT '',  -- For feature drift, name of the drifted feature

    -- Severity
    severity Enum8(
        'low' = 1,
        'medium' = 2,
        'high' = 3,
        'critical' = 4
    ),

    -- Scores
    drift_score Float64,
    p_value Float64 DEFAULT 1.0,
    threshold Float64,

    -- Time windows
    reference_window_start DateTime64(3),
    reference_window_end DateTime64(3),
    test_window_start DateTime64(3),
    test_window_end DateTime64(3),

    -- Sample counts
    reference_sample_count UInt64 DEFAULT 0,
    test_sample_count UInt64 DEFAULT 0,

    -- Detailed metrics (JSON object)
    metrics String DEFAULT '{}',

    -- Affected data slices (JSON array)
    affected_slices String DEFAULT '[]',

    -- Human-readable explanation
    explanation String DEFAULT '',

    -- Alert status
    status Enum8(
        'open' = 1,
        'acknowledged' = 2,
        'resolved' = 3,
        'ignored' = 4
    ) DEFAULT 'open',

    -- Resolution details
    resolved_at Nullable(DateTime64(3)),
    resolved_by String DEFAULT '',
    resolution_notes String DEFAULT '',

    -- Service information
    service_name LowCardinality(String) DEFAULT '',
    environment LowCardinality(String) DEFAULT 'development',

    -- Partitioning
    event_date Date DEFAULT toDate(detected_at),

    -- Timestamps
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(event_date)
ORDER BY (model_id, drift_type, alert_id)
TTL event_date + INTERVAL 365 DAY
SETTINGS index_granularity = 8192;

-- Indices
ALTER TABLE pyflare.drift_alerts
    ADD INDEX idx_status status TYPE set(4) GRANULARITY 4,
    ADD INDEX idx_severity severity TYPE set(4) GRANULARITY 4;

-- Drift reference distributions table
-- Stores baseline distributions for comparison
CREATE TABLE IF NOT EXISTS pyflare.drift_references
(
    reference_id UUID DEFAULT generateUUIDv4(),

    -- What this reference is for
    model_id LowCardinality(String),
    feature_name String,
    drift_type Enum8(
        'feature' = 1,
        'embedding' = 2,
        'concept' = 3,
        'prediction' = 4
    ),

    -- Time window of the reference data
    window_start DateTime64(3),
    window_end DateTime64(3),
    sample_count UInt64,

    -- Statistical summary (for feature drift)
    mean Float64 DEFAULT 0.0,
    std_dev Float64 DEFAULT 0.0,
    min_value Float64 DEFAULT 0.0,
    max_value Float64 DEFAULT 0.0,
    percentiles Array(Float64) DEFAULT [],  -- [p25, p50, p75, p90, p95, p99]

    -- Distribution data (JSON or binary serialized)
    distribution_data String DEFAULT '',
    distribution_format LowCardinality(String) DEFAULT 'json',  -- 'json', 'protobuf', 'numpy'

    -- For embedding drift - centroid vector
    centroid Array(Float32) DEFAULT [],

    -- Metadata
    is_active UInt8 DEFAULT 1,
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (model_id, feature_name, drift_type, window_start)
SETTINGS index_granularity = 8192;

-- Materialized view for drift alert counts by model
CREATE MATERIALIZED VIEW IF NOT EXISTS pyflare.mv_drift_alert_counts
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (model_id, drift_type, severity, event_date)
AS SELECT
    event_date,
    model_id,
    drift_type,
    severity,
    count() AS alert_count,
    countIf(status = 'open') AS open_count,
    countIf(status = 'resolved') AS resolved_count
FROM pyflare.drift_alerts
GROUP BY event_date, model_id, drift_type, severity;

-- View for active (open) alerts
CREATE VIEW IF NOT EXISTS pyflare.active_drift_alerts AS
SELECT *
FROM pyflare.drift_alerts
WHERE status = 'open'
ORDER BY severity DESC, detected_at DESC;
