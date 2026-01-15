# PyFlare Configuration Guide

This guide covers all configuration options for both the PyFlare Collector and Python SDK.

## Table of Contents

- [Collector Configuration](#collector-configuration)
- [SDK Configuration](#sdk-configuration)
- [Environment Variables](#environment-variables)
- [Production Recommendations](#production-recommendations)

---

## Collector Configuration

The PyFlare Collector is configured via a YAML file. By default, it looks for `config/collector.yaml`.

### Complete Configuration Reference

```yaml
# PyFlare Collector Configuration

# =============================================================================
# OTLP Receiver Configuration
# =============================================================================
receiver:
  # gRPC receiver settings
  grpc:
    # Listen address (host:port)
    endpoint: "0.0.0.0:4317"

    # Maximum message size (bytes)
    max_recv_msg_size_bytes: 16777216  # 16 MB

    # Maximum concurrent streams
    max_concurrent_streams: 100

    # Enable gRPC reflection (useful for debugging)
    enable_reflection: true

    # Keepalive settings
    keepalive:
      time_seconds: 30        # Send keepalive pings every 30s
      timeout_seconds: 10     # Wait 10s for ping response
      permit_without_stream: true

  # HTTP receiver settings
  http:
    # Listen address (host:port)
    endpoint: "0.0.0.0:4318"

    # Maximum request body size (bytes)
    max_request_body_bytes: 16777216  # 16 MB

    # Timeouts
    read_timeout_seconds: 30
    write_timeout_seconds: 30

    # CORS configuration
    cors_allowed_origins:
      - "*"  # Allow all origins (restrict in production)

# =============================================================================
# Batching Configuration
# =============================================================================
batcher:
  # Maximum spans per batch
  max_batch_size: 512

  # Maximum time to wait before flushing (milliseconds)
  max_batch_timeout_ms: 5000

  # Maximum queue size (backpressure threshold)
  max_queue_size: 10000

  # Number of worker threads
  num_workers: 4

# =============================================================================
# Sampling Configuration
# =============================================================================
sampler:
  # Sampling strategy
  # Options: always_on, always_off, probabilistic, rate_limiting, parent_based
  strategy: probabilistic

  # For probabilistic sampling: sample rate (0.0 to 1.0)
  probability: 1.0

  # For rate_limiting sampling: traces per second
  traces_per_second: 100

  # Per-service sampling rates (optional)
  # Overrides default for specific services
  service_rates:
    # Sample 100% of critical-service traces
    critical-service: 1.0
    # Sample only 10% of high-volume-service traces
    high-volume-service: 0.1

# =============================================================================
# Kafka Export Configuration
# =============================================================================
kafka:
  # Broker addresses
  brokers:
    - "localhost:9092"
    # - "kafka-2:9092"  # Additional brokers for HA

  # Topic names
  topics:
    traces: "pyflare.traces"
    metrics: "pyflare.metrics"
    logs: "pyflare.logs"

  # Producer settings
  producer:
    # Batch size in bytes
    batch_size: 16384

    # Time to wait for batch to fill (milliseconds)
    linger_ms: 5

    # Compression algorithm: none, gzip, snappy, lz4, zstd
    compression: "lz4"

    # Acknowledgment level: 0, 1, all
    acks: "all"

    # Retry settings
    retries: 3
    retry_backoff_ms: 100

    # Enable exactly-once semantics
    enable_idempotence: true

    # Max in-flight requests
    max_in_flight_requests: 5

  # Timeout settings
  timeouts:
    message_timeout_ms: 30000
    socket_timeout_ms: 60000
    metadata_timeout_ms: 10000

  # Security settings (optional)
  security:
    # Protocol: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    protocol: "PLAINTEXT"

    # SSL configuration (when using SSL or SASL_SSL)
    ssl:
      ca_location: "/path/to/ca.pem"
      cert_location: "/path/to/client.pem"
      key_location: "/path/to/client.key"
      key_password: ""

    # SASL configuration (when using SASL_PLAINTEXT or SASL_SSL)
    sasl:
      mechanism: "SCRAM-SHA-256"  # PLAIN, SCRAM-SHA-256, SCRAM-SHA-512
      username: "your-username"
      password: "your-password"

# =============================================================================
# General Settings
# =============================================================================
general:
  # Number of worker threads for processing
  worker_threads: 8

  # Health check endpoint
  health_endpoint: "0.0.0.0:8081"

  # Metrics endpoint path
  metrics_path: "/metrics"

  # Service name (for internal telemetry)
  service_name: "pyflare-collector"

  # Enable Prometheus metrics
  enable_metrics: true

# =============================================================================
# Enrichment Settings
# =============================================================================
enrichment:
  # Add host information to spans
  add_host_info: true

  # Add process information to spans
  add_process_info: true

  # Normalize timestamps to UTC
  normalize_timestamps: true

  # Custom attributes added to all spans
  custom_attributes:
    environment: "production"
    datacenter: "us-west-2"
    team: "ml-platform"
```

### Minimal Configuration

For development, you can use a minimal configuration:

```yaml
receiver:
  grpc:
    endpoint: "0.0.0.0:4317"
  http:
    endpoint: "0.0.0.0:4318"

batcher:
  max_batch_size: 100
  max_batch_timeout_ms: 1000

sampler:
  strategy: always_on

# No Kafka - spans are logged to console
```

### CLI Options

The collector supports command-line overrides:

```bash
pyflare_collector \
  --config /path/to/config.yaml \
  --grpc-endpoint 0.0.0.0:4317 \
  --http-endpoint 0.0.0.0:4318 \
  --log-level debug \
  --kafka-brokers kafka-1:9092,kafka-2:9092 \
  --sample-rate 0.5 \
  --batch-size 256
```

CLI options override config file values.

---

## SDK Configuration

### Programmatic Configuration

```python
import pyflare

pyflare.init(
    # Required
    service_name="my-ml-service",

    # Connection
    endpoint="http://localhost:4317",  # Collector endpoint
    headers={"Authorization": "Bearer token"},  # Auth headers
    use_http=False,  # Use gRPC (default) or HTTP

    # Service metadata
    environment="production",
    version="1.2.3",

    # Behavior
    enabled=True,          # Enable/disable tracing
    sample_rate=1.0,       # Sample rate (0.0-1.0)
    batch_export=True,     # Batch spans (recommended)
    debug=False,           # Print spans to console

    # Custom resource attributes
    resource_attributes={
        "deployment.region": "us-west-2",
        "team": "ml-platform",
        "k8s.pod.name": os.environ.get("HOSTNAME", ""),
    },
)
```

### Configuration Patterns

#### Development Configuration

```python
pyflare.init(
    service_name="my-service",
    endpoint="http://localhost:4317",
    environment="development",
    debug=True,            # See spans in console
    sample_rate=1.0,       # Trace everything
    batch_export=False,    # Immediate export for debugging
)
```

#### Production Configuration

```python
pyflare.init(
    service_name="my-service",
    endpoint="http://collector.internal:4317",
    environment="production",
    version=os.environ.get("APP_VERSION", "unknown"),
    sample_rate=0.1,       # Sample 10% in production
    batch_export=True,     # Efficient batching
    resource_attributes={
        "k8s.namespace": os.environ.get("K8S_NAMESPACE"),
        "k8s.pod.name": os.environ.get("HOSTNAME"),
    },
)
```

#### Disabled Configuration

```python
# Completely disable tracing
pyflare.init(
    service_name="my-service",
    enabled=False,
)

# Or via environment variable
# PYFLARE_ENABLED=false
```

---

## Environment Variables

PyFlare supports configuration via environment variables. These override programmatic settings.

### SDK Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PYFLARE_ENDPOINT` | Collector endpoint | `http://collector:4317` |
| `PYFLARE_ENABLED` | Enable/disable tracing | `true`, `false` |
| `PYFLARE_SAMPLE_RATE` | Sampling rate | `0.5` |
| `PYFLARE_ENVIRONMENT` | Deployment environment | `production` |

### Collector Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PYFLARE_LOG_LEVEL` | Log level | `debug`, `info`, `warn`, `error` |
| `PYFLARE_GRPC_ENDPOINT` | Override gRPC endpoint | `0.0.0.0:4317` |
| `PYFLARE_HTTP_ENDPOINT` | Override HTTP endpoint | `0.0.0.0:4318` |
| `PYFLARE_KAFKA_BROKERS` | Kafka brokers (comma-separated) | `kafka-1:9092,kafka-2:9092` |
| `PYFLARE_SAMPLE_RATE` | Override sample rate | `0.5` |

### Using Environment Variables

```bash
# In your shell or Docker environment
export PYFLARE_ENDPOINT="http://collector.production:4317"
export PYFLARE_ENABLED="true"
export PYFLARE_SAMPLE_RATE="0.1"
export PYFLARE_ENVIRONMENT="production"
```

```python
# SDK automatically picks up environment variables
import pyflare

pyflare.init(
    service_name="my-service",
    # endpoint, enabled, sample_rate, environment are read from env vars
)
```

---

## Production Recommendations

### Sampling Strategy

For high-traffic services, use sampling to control costs:

```yaml
sampler:
  strategy: probabilistic
  probability: 0.1  # Sample 10% of traces

  # But sample 100% of critical services
  service_rates:
    payment-service: 1.0
    auth-service: 1.0
```

### Batching Configuration

Optimize batching for your workload:

```yaml
batcher:
  # High throughput: larger batches, longer timeout
  max_batch_size: 1000
  max_batch_timeout_ms: 10000

  # Low latency: smaller batches, shorter timeout
  # max_batch_size: 100
  # max_batch_timeout_ms: 1000

  # Backpressure: increase queue size for bursts
  max_queue_size: 50000

  # Scale workers with CPU cores
  num_workers: 8
```

### Kafka Configuration

Production Kafka settings:

```yaml
kafka:
  brokers:
    - "kafka-1.internal:9092"
    - "kafka-2.internal:9092"
    - "kafka-3.internal:9092"

  producer:
    compression: "lz4"      # Good balance of speed/compression
    acks: "all"             # Durability
    enable_idempotence: true
    retries: 5

  security:
    protocol: "SASL_SSL"
    sasl:
      mechanism: "SCRAM-SHA-256"
      username: "${KAFKA_USERNAME}"
      password: "${KAFKA_PASSWORD}"
```

### Resource Allocation

Docker/Kubernetes resources:

```yaml
# Kubernetes example
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Health Checks

Configure health checks for orchestration:

```yaml
# docker-compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:4318/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Monitoring

Enable metrics endpoint for Prometheus:

```yaml
general:
  enable_metrics: true
  health_endpoint: "0.0.0.0:8081"
  metrics_path: "/metrics"
```

Scrape with Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'pyflare-collector'
    static_configs:
      - targets: ['collector:8081']
```

### Security Considerations

1. **Use TLS for gRPC/HTTP endpoints** in production
2. **Authenticate Kafka connections** with SASL
3. **Limit CORS origins** for HTTP endpoint
4. **Run as non-root user** in containers
5. **Use secrets management** for credentials

```yaml
# Don't put credentials in config files
security:
  sasl:
    username: "${KAFKA_USERNAME}"  # From environment
    password: "${KAFKA_PASSWORD}"
```

---

## Phase 2: Advanced Configuration

### Drift Detection Configuration

Configure drift detection algorithms and thresholds:

```yaml
# =============================================================================
# Drift Detection Configuration
# =============================================================================
drift:
  # Enable drift detection
  enabled: true

  # PSI (Population Stability Index) settings
  psi:
    threshold: 0.2           # PSI > 0.2 indicates significant drift
    num_bins: 10             # Number of bins for discretization
    min_samples_per_bin: 5   # Minimum samples for reliable calculation

  # MMD (Maximum Mean Discrepancy) settings
  mmd:
    threshold: 0.1           # MMD threshold for drift detection
    rbf_sigma: 0.0           # RBF kernel bandwidth (0 = auto via median heuristic)
    num_permutations: 100    # Permutations for p-value estimation
    p_value_threshold: 0.05  # Statistical significance level
    max_samples: 1000        # Max samples for performance

  # KS Test settings
  ks:
    p_value_threshold: 0.05  # P-value threshold for drift detection

  # Reference management
  reference:
    # Storage backend: qdrant, redis, or memory
    storage: qdrant
    qdrant:
      collection: "pyflare_references"
      endpoint: "http://qdrant:6333"
    # How often to update reference (0 = manual only)
    auto_update_interval_hours: 24
    # Minimum samples before updating reference
    min_samples_for_update: 1000

  # Alert configuration
  alerts:
    enabled: true
    # Alert on these drift types
    types:
      - feature
      - embedding
      - prediction
    # Severity thresholds
    severity:
      low: 0.1
      medium: 0.2
      high: 0.3
    # Webhook for alerts
    webhook_url: "${DRIFT_WEBHOOK_URL}"
```

### Evaluation Configuration

Configure LLM evaluation pipelines:

```yaml
# =============================================================================
# Evaluation Configuration
# =============================================================================
evaluation:
  # Enable automatic evaluation
  enabled: true

  # LLM Judge settings
  llm_judge:
    enabled: true
    api_endpoint: "https://api.openai.com/v1/chat/completions"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 1024
    timeout_seconds: 30
    max_retries: 3
    # Cache judge results
    enable_cache: true
    cache_ttl_hours: 24

  # RAG evaluation settings
  rag:
    enabled: true
    # Use embeddings for similarity (requires API key)
    use_embedding_similarity: false
    # Thresholds for pass/fail
    faithfulness_threshold: 0.7
    context_relevance_threshold: 0.5
    # Weights for overall score
    weights:
      context_relevance: 0.25
      faithfulness: 0.35
      answer_relevance: 0.20
      groundedness: 0.20

  # Toxicity detection settings
  toxicity:
    enabled: true
    # Categories to detect
    categories:
      - hate_speech
      - harassment
      - violence
      - sexual_content
      - self_harm
    # Threshold for flagging
    threshold: 0.5
    # Custom blocked words
    custom_blocked_words: []

  # PII detection settings
  pii:
    enabled: true
    # PII types to detect
    types:
      - email
      - phone
      - ssn
      - credit_card
      - ip_address
    # Action: detect, scrub, or block
    action: detect
    # Replacement for scrubbed PII
    scrub_replacement: "[REDACTED]"
```

### Budget Management Configuration

Configure cost budgets and alerts:

```yaml
# =============================================================================
# Budget Management Configuration
# =============================================================================
budget:
  # Enable budget management
  enabled: true

  # Redis backend for budget tracking
  redis:
    endpoint: "redis:6379"
    key_prefix: "pyflare:budget"
    # Sync interval for local counters
    sync_interval_seconds: 5

  # Default budgets
  defaults:
    # Global daily budget
    global:
      daily_limit_micros: 100000000  # $100
      warning_threshold: 0.8
      block_on_exceeded: false

    # Per-user defaults
    user:
      daily_limit_micros: 10000000   # $10
      warning_threshold: 0.8
      block_on_exceeded: true

    # Per-model defaults
    model:
      daily_limit_micros: 50000000   # $50
      warning_threshold: 0.9
      block_on_exceeded: false

  # Alert configuration
  alerts:
    enabled: true
    # Webhook for budget alerts
    webhook_url: "${BUDGET_WEBHOOK_URL}"
    # Alert channels
    channels:
      - webhook
      - email
    email:
      recipients:
        - "alerts@example.com"
      smtp_host: "${SMTP_HOST}"
      smtp_port: 587
```

### Root Cause Analysis Configuration

Configure RCA pipelines:

```yaml
# =============================================================================
# Root Cause Analysis Configuration
# =============================================================================
rca:
  # Enable RCA
  enabled: true

  # Slice analysis settings
  slice_analyzer:
    # Minimum samples for a slice to be analyzed
    min_samples: 100
    # Maximum slices to return
    max_slices: 50
    # Deviation threshold for reporting (percentage)
    deviation_threshold: 0.1
    # P-value threshold for significance
    p_value_threshold: 0.05
    # Analysis time window
    analysis_window_hours: 24
    # Dimensions to analyze automatically
    auto_dimensions:
      - model
      - user
      - feature
      - input_length
      - time_of_day

  # Pattern detection settings
  pattern_detector:
    # Minimum support for pattern detection
    min_support: 0.1
    # Lookback window for temporal patterns
    lookback_hours: 24
    # Pattern types to detect
    pattern_types:
      - error_spike
      - latency_degradation
      - quality_drop
      - cost_anomaly
      - drift_correlation

  # Failure clustering settings
  failure_clusterer:
    # Clustering method: text_similarity, embedding, dbscan
    method: text_similarity
    # Minimum cluster size
    min_cluster_size: 5
    # Similarity threshold for text clustering
    similarity_threshold: 0.7

  # Storage
  storage:
    # ClickHouse for analytics
    clickhouse:
      endpoint: "http://clickhouse:8123"
      database: "pyflare"
```

### Storage Backend Configuration

Configure Phase 2 storage backends:

```yaml
# =============================================================================
# Storage Configuration
# =============================================================================
storage:
  # ClickHouse for OLAP queries
  clickhouse:
    endpoint: "http://clickhouse:8123"
    database: "pyflare"
    username: "${CLICKHOUSE_USER}"
    password: "${CLICKHOUSE_PASSWORD}"
    # Connection pool
    max_connections: 10
    connection_timeout_seconds: 30
    # Batch insert settings
    batch_size: 10000
    flush_interval_seconds: 5

  # Qdrant for vector storage
  qdrant:
    endpoint: "http://qdrant:6333"
    api_key: "${QDRANT_API_KEY}"
    # Collection settings
    collections:
      embeddings:
        name: "pyflare_embeddings"
        vector_size: 1536
        distance: "Cosine"
      references:
        name: "pyflare_references"
        vector_size: 1536
        distance: "Cosine"

  # Redis for caching and counters
  redis:
    endpoint: "redis:6379"
    password: "${REDIS_PASSWORD}"
    database: 0
    # Connection pool
    max_connections: 50
    # Key prefixes
    prefixes:
      budget: "pyflare:budget"
      cache: "pyflare:cache"
      rate_limit: "pyflare:ratelimit"
```

### Query API Configuration

Configure the Query API server:

```yaml
# =============================================================================
# Query API Configuration
# =============================================================================
query_api:
  # Server settings
  endpoint: "0.0.0.0:8080"

  # CORS
  cors:
    allowed_origins:
      - "http://localhost:3000"
      - "https://dashboard.example.com"
    allowed_methods:
      - GET
      - POST
      - PUT
      - DELETE
    allowed_headers:
      - Authorization
      - Content-Type

  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_second: 100
    burst: 200

  # Authentication
  auth:
    enabled: false
    # JWT settings
    jwt:
      secret: "${JWT_SECRET}"
      issuer: "pyflare"
      expiry_hours: 24

  # Pagination defaults
  pagination:
    default_limit: 100
    max_limit: 1000
```

### SDK Phase 2 Configuration

Additional SDK configuration for Phase 2 features:

```python
import pyflare

pyflare.init(
    service_name="my-service",
    endpoint="http://localhost:4317",

    # Drift detection
    drift_detection={
        "enabled": True,
        "types": ["feature", "embedding"],
        "threshold": 0.1,
    },

    # Evaluations
    evaluations={
        "enabled": True,
        "types": ["hallucination", "toxicity"],
        "llm_judge": {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        },
    },

    # Budget
    budget={
        "enabled": True,
        "daily_limit_usd": 100.0,
        "warning_threshold": 0.8,
        "block_on_exceeded": False,
    },
)
```

### Environment Variables (Phase 2)

Additional environment variables for Phase 2:

| Variable | Description | Example |
|----------|-------------|---------|
| `PYFLARE_DRIFT_ENABLED` | Enable drift detection | `true` |
| `PYFLARE_DRIFT_THRESHOLD` | Drift detection threshold | `0.1` |
| `PYFLARE_EVAL_ENABLED` | Enable evaluations | `true` |
| `PYFLARE_EVAL_API_KEY` | API key for LLM judge | `sk-...` |
| `PYFLARE_BUDGET_ENABLED` | Enable budget management | `true` |
| `PYFLARE_BUDGET_DAILY_LIMIT` | Daily budget limit (USD) | `100` |
| `QDRANT_ENDPOINT` | Qdrant vector DB endpoint | `http://qdrant:6333` |
| `CLICKHOUSE_ENDPOINT` | ClickHouse endpoint | `http://clickhouse:8123` |
| `REDIS_ENDPOINT` | Redis endpoint | `redis:6379` |
