# PyFlare Component Guide

This guide provides detailed documentation for each component in PyFlare's Phase 1 implementation. Use this as a reference when understanding, debugging, or extending the codebase.

## Table of Contents

- [OTLP Receiver](#otlp-receiver)
- [Sampler](#sampler)
- [Batcher](#batcher)
- [Kafka Exporter](#kafka-exporter)
- [Python SDK](#python-sdk)
- [Cost Calculator](#cost-calculator)

---

## OTLP Receiver

**Location:** `src/collector/otlp_receiver.cpp`

The OTLP Receiver accepts telemetry data via gRPC and HTTP protocols.

### Class Structure

```cpp
class OtlpReceiver {
public:
    explicit OtlpReceiver(const ReceiverConfig& config);
    ~OtlpReceiver();

    // Lifecycle
    absl::Status Start();
    void Shutdown();

    // Callback registration
    void OnSpans(SpanCallback callback);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

### Implementation Details

#### gRPC Service (Conditional)

When `PYFLARE_HAS_GRPC` is defined, the receiver implements the OTLP TraceService:

```cpp
#ifdef PYFLARE_HAS_GRPC
class TraceServiceImpl final : public pyflare::v1::TraceService::Service {
public:
    grpc::Status Export(
        grpc::ServerContext* context,
        const pyflare::v1::ExportTraceServiceRequest* request,
        pyflare::v1::ExportTraceServiceResponse* response) override {

        // Parse spans from protobuf
        std::vector<Span> spans = ParseProtoSpans(request);

        // Forward to callback
        if (span_callback_) {
            span_callback_(std::move(spans));
        }

        return grpc::Status::OK;
    }
};
#endif
```

#### HTTP Endpoint (Conditional)

When `PYFLARE_HAS_HTTPLIB` is defined:

```cpp
#ifdef PYFLARE_HAS_HTTPLIB
void SetupHttpServer() {
    http_server_.Post("/v1/traces", [this](const httplib::Request& req,
                                            httplib::Response& res) {
        // Parse JSON body
        auto json = nlohmann::json::parse(req.body);
        std::vector<Span> spans = ParseJsonSpans(json);

        // Forward to callback
        if (span_callback_) {
            span_callback_(std::move(spans));
        }

        res.status = 200;
        res.set_content("{}", "application/json");
    });

    // Health endpoint
    http_server_.Get("/health", [](const httplib::Request&,
                                   httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
}
#endif
```

#### Span Parsing

The receiver parses OTLP spans into internal format:

```cpp
Span ParseSpan(const ResourceSpan& resource_span,
               const ScopeSpan& scope_span,
               const ProtoSpan& proto_span) {
    Span span;

    // Core identifiers
    span.trace_id = BytesToHex(proto_span.trace_id());
    span.span_id = BytesToHex(proto_span.span_id());
    span.parent_span_id = BytesToHex(proto_span.parent_span_id());

    // Metadata
    span.name = proto_span.name();
    span.kind = ConvertSpanKind(proto_span.kind());
    span.start_time_unix_nano = proto_span.start_time_unix_nano();
    span.end_time_unix_nano = proto_span.end_time_unix_nano();

    // Status
    span.status_code = ConvertStatusCode(proto_span.status().code());
    span.status_message = proto_span.status().message();

    // Attributes
    for (const auto& attr : proto_span.attributes()) {
        span.attributes[attr.key()] = ConvertAttributeValue(attr.value());
    }

    // Resource attributes
    for (const auto& attr : resource_span.resource().attributes()) {
        if (attr.key() == "service.name") {
            span.service_name = attr.value().string_value();
        }
        span.resource_attributes[attr.key()] = attr.value().string_value();
    }

    return span;
}
```

### Configuration

```yaml
receiver:
  grpc:
    endpoint: "0.0.0.0:4317"
    max_recv_msg_size_bytes: 16777216
    max_concurrent_streams: 100
    enable_reflection: true
    keepalive:
      time_seconds: 30
      timeout_seconds: 10

  http:
    endpoint: "0.0.0.0:4318"
    max_request_body_bytes: 16777216
    read_timeout_seconds: 30
    cors_allowed_origins:
      - "*"
```

### Thread Safety

- Callback invocations are protected by mutex
- gRPC handles threading internally
- HTTP server runs in dedicated thread

---

## Sampler

**Location:** `src/collector/sampler.cpp`

The Sampler decides which spans to keep based on configurable strategies.

### Class Structure

```cpp
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config);
    ~Sampler();

    // Main sampling method
    bool ShouldSample(const Span& span) const;

    // Statistics
    uint64_t GetSampledCount() const;
    uint64_t GetDroppedCount() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

### Sampling Strategies

#### AlwaysOn / AlwaysOff

Simplest strategies - always keep or always drop:

```cpp
bool AlwaysOnSampler::ShouldSample(const Span&) const {
    return true;
}

bool AlwaysOffSampler::ShouldSample(const Span&) const {
    return false;
}
```

#### Probabilistic Sampling

Uses FNV-1a hash for deterministic, trace-consistent sampling:

```cpp
class ProbabilisticSampler {
    double probability_;
    uint64_t threshold_;

public:
    ProbabilisticSampler(double probability)
        : probability_(probability)
        , threshold_(static_cast<uint64_t>(probability * UINT64_MAX)) {}

    bool ShouldSample(const Span& span) const {
        // FNV-1a hash of trace ID
        uint64_t hash = FNV1aHash(span.trace_id);
        return hash <= threshold_;
    }
};

uint64_t FNV1aHash(const std::string& data) {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;

    uint64_t hash = FNV_OFFSET;
    for (char c : data) {
        hash ^= static_cast<uint64_t>(c);
        hash *= FNV_PRIME;
    }
    return hash;
}
```

**Why FNV-1a?**
- Fast computation
- Good distribution
- Deterministic: same trace ID always produces same decision
- All spans in a trace get the same decision

#### Rate-Limiting Sampling

Uses token bucket algorithm:

```cpp
class RateLimitingSampler {
    double traces_per_second_;
    mutable std::atomic<double> tokens_;
    mutable std::chrono::steady_clock::time_point last_update_;
    mutable std::mutex mutex_;

public:
    bool ShouldSample(const Span& span) const {
        std::lock_guard<std::mutex> lock(mutex_);

        // Refill tokens
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_update_).count();
        tokens_ = std::min(traces_per_second_, tokens_ + elapsed * traces_per_second_);
        last_update_ = now;

        // Try to consume a token
        if (tokens_ >= 1.0) {
            tokens_ -= 1.0;
            return true;
        }
        return false;
    }
};
```

#### Parent-Based Sampling

Respects the sampling decision of the parent span:

```cpp
bool ParentBasedSampler::ShouldSample(const Span& span) const {
    // Root spans: delegate to configured root sampler
    if (span.parent_span_id.empty()) {
        return root_sampler_->ShouldSample(span);
    }

    // Non-root spans: check parent's sampling flag
    // This requires trace context propagation
    auto it = span.attributes.find("pyflare.parent.sampled");
    if (it != span.attributes.end()) {
        return std::get<bool>(it->second);
    }

    // Default: sample if parent was sampled (assume true if unknown)
    return true;
}
```

#### Composite Sampling

Per-service sampling rates:

```cpp
class CompositeSampler {
    std::unordered_map<std::string, double> service_rates_;
    std::unique_ptr<Sampler> default_sampler_;

public:
    bool ShouldSample(const Span& span) const {
        // Check for service-specific rate
        auto it = service_rates_.find(span.service_name);
        if (it != service_rates_.end()) {
            return ProbabilisticSample(span.trace_id, it->second);
        }

        // Fall back to default sampler
        return default_sampler_->ShouldSample(span);
    }
};
```

### Configuration

```yaml
sampler:
  strategy: probabilistic  # always_on, always_off, probabilistic, rate_limiting, parent_based
  probability: 0.1         # For probabilistic
  traces_per_second: 100   # For rate_limiting

  # Per-service overrides
  service_rates:
    critical-service: 1.0
    high-volume-service: 0.01
```

---

## Batcher

**Location:** `src/collector/batcher.cpp`

The Batcher accumulates spans into batches for efficient export.

### Class Structure

```cpp
class Batcher {
public:
    explicit Batcher(const BatcherConfig& config);
    ~Batcher();

    // Add spans to queue
    void Add(std::vector<Span>&& spans);
    void Add(Span&& span);

    // Lifecycle
    void Start();
    void Shutdown();

    // Callback registration
    void OnBatch(BatchCallback callback);

    // Statistics
    size_t GetPendingCount() const;
    size_t GetBatchesSent() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

### Implementation Details

#### Queue Management

```cpp
class Batcher::Impl {
    std::deque<Span> pending_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    size_t max_batch_size_;
    size_t max_queue_size_;
    std::chrono::milliseconds batch_timeout_;

    std::atomic<bool> running_{false};
    std::thread timer_thread_;
    std::vector<std::thread> worker_threads_;
};
```

#### Adding Spans

```cpp
void Batcher::Impl::Add(std::vector<Span>&& spans) {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // Backpressure: if queue is full, wait or drop
    if (pending_queue_.size() >= max_queue_size_) {
        spdlog::warn("Batcher queue full, applying backpressure");

        // Option 1: Block until space available
        queue_cv_.wait(lock, [this] {
            return pending_queue_.size() < max_queue_size_ || !running_;
        });

        // Option 2: Drop oldest spans (uncomment to enable)
        // while (pending_queue_.size() >= max_queue_size_) {
        //     pending_queue_.pop_front();
        //     dropped_count_++;
        // }
    }

    // Add spans to queue
    for (auto& span : spans) {
        pending_queue_.push_back(std::move(span));
    }

    // Check if batch is ready
    if (pending_queue_.size() >= max_batch_size_) {
        queue_cv_.notify_one();
    }
}
```

#### Timer Thread

```cpp
void Batcher::Impl::TimerLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Wait for timeout or notification
        bool notified = queue_cv_.wait_for(lock, batch_timeout_, [this] {
            return pending_queue_.size() >= max_batch_size_ || !running_;
        });

        if (!running_) break;

        // Flush if we have any spans (timeout or batch full)
        if (!pending_queue_.empty()) {
            FlushBatch(lock);
        }
    }
}
```

#### Worker Threads

```cpp
void Batcher::Impl::WorkerLoop() {
    while (running_) {
        std::vector<Span> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait for work
            queue_cv_.wait(lock, [this] {
                return !batch_ready_queue_.empty() || !running_;
            });

            if (!running_ && batch_ready_queue_.empty()) break;

            // Get batch from ready queue
            if (!batch_ready_queue_.empty()) {
                batch = std::move(batch_ready_queue_.front());
                batch_ready_queue_.pop();
            }
        }

        // Process batch outside lock
        if (!batch.empty() && batch_callback_) {
            batch_callback_(std::move(batch));
        }
    }
}

void Batcher::Impl::FlushBatch(std::unique_lock<std::mutex>& lock) {
    std::vector<Span> batch;
    batch.reserve(max_batch_size_);

    // Extract up to max_batch_size spans
    size_t count = std::min(pending_queue_.size(), max_batch_size_);
    for (size_t i = 0; i < count; ++i) {
        batch.push_back(std::move(pending_queue_.front()));
        pending_queue_.pop_front();
    }

    // Add to ready queue for workers
    batch_ready_queue_.push(std::move(batch));
    queue_cv_.notify_one();

    batches_sent_++;
}
```

### Configuration

```yaml
batcher:
  max_batch_size: 512           # Spans per batch
  max_batch_timeout_ms: 5000    # Max wait time
  max_queue_size: 10000         # Backpressure threshold
  num_workers: 4                # Worker threads
```

### Tuning Guide

| Scenario | `max_batch_size` | `timeout_ms` | `num_workers` |
|----------|------------------|--------------|---------------|
| High throughput | 1000 | 10000 | 8 |
| Low latency | 100 | 1000 | 4 |
| Memory constrained | 256 | 5000 | 2 |
| Development | 10 | 1000 | 1 |

---

## Kafka Exporter

**Location:** `src/collector/kafka_exporter.cpp`

The Kafka Exporter publishes span batches to Kafka topics.

### Class Structure

```cpp
class KafkaExporter {
public:
    explicit KafkaExporter(const KafkaConfig& config);
    ~KafkaExporter();

    // Lifecycle
    absl::Status Connect();
    void Disconnect();

    // Export
    absl::Status Export(const std::vector<Span>& spans);

    // Statistics
    uint64_t GetMessagesSent() const;
    uint64_t GetMessagesDelivered() const;
    uint64_t GetMessagesFailed() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

### Implementation Details

#### Producer Setup (Conditional)

```cpp
#ifdef PYFLARE_HAS_RDKAFKA
absl::Status KafkaExporter::Impl::Connect() {
    // Create configuration
    auto conf = std::unique_ptr<rd_kafka_conf_t, decltype(&rd_kafka_conf_destroy)>(
        rd_kafka_conf_new(), rd_kafka_conf_destroy);

    // Set brokers
    rd_kafka_conf_set(conf.get(), "bootstrap.servers",
                      config_.brokers_string().c_str(), nullptr, 0);

    // Producer settings
    rd_kafka_conf_set(conf.get(), "batch.size",
                      std::to_string(config_.producer.batch_size).c_str(), nullptr, 0);
    rd_kafka_conf_set(conf.get(), "linger.ms",
                      std::to_string(config_.producer.linger_ms).c_str(), nullptr, 0);
    rd_kafka_conf_set(conf.get(), "compression.type",
                      config_.producer.compression.c_str(), nullptr, 0);
    rd_kafka_conf_set(conf.get(), "acks",
                      config_.producer.acks.c_str(), nullptr, 0);
    rd_kafka_conf_set(conf.get(), "enable.idempotence",
                      config_.producer.enable_idempotence ? "true" : "false", nullptr, 0);

    // Security (if configured)
    if (config_.security.protocol != "PLAINTEXT") {
        rd_kafka_conf_set(conf.get(), "security.protocol",
                          config_.security.protocol.c_str(), nullptr, 0);

        if (config_.security.protocol.find("SSL") != std::string::npos) {
            rd_kafka_conf_set(conf.get(), "ssl.ca.location",
                              config_.security.ssl.ca_location.c_str(), nullptr, 0);
            // ... more SSL settings
        }

        if (config_.security.protocol.find("SASL") != std::string::npos) {
            rd_kafka_conf_set(conf.get(), "sasl.mechanism",
                              config_.security.sasl.mechanism.c_str(), nullptr, 0);
            rd_kafka_conf_set(conf.get(), "sasl.username",
                              config_.security.sasl.username.c_str(), nullptr, 0);
            rd_kafka_conf_set(conf.get(), "sasl.password",
                              config_.security.sasl.password.c_str(), nullptr, 0);
        }
    }

    // Set delivery callback
    rd_kafka_conf_set_dr_msg_cb(conf.get(), DeliveryCallback);

    // Create producer
    char errstr[512];
    producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf.release(), errstr, sizeof(errstr));
    if (!producer_) {
        return absl::InternalError(absl::StrCat("Failed to create Kafka producer: ", errstr));
    }

    return absl::OkStatus();
}
#endif
```

#### Message Serialization

```cpp
std::string SerializeSpansToJson(const std::vector<Span>& spans) {
    nlohmann::json json_array = nlohmann::json::array();

    for (const auto& span : spans) {
        nlohmann::json span_json = {
            {"trace_id", span.trace_id},
            {"span_id", span.span_id},
            {"parent_span_id", span.parent_span_id},
            {"name", span.name},
            {"kind", static_cast<int>(span.kind)},
            {"start_time_unix_nano", span.start_time_unix_nano},
            {"end_time_unix_nano", span.end_time_unix_nano},
            {"status_code", static_cast<int>(span.status_code)},
            {"status_message", span.status_message},
            {"service_name", span.service_name},
        };

        // Add attributes
        nlohmann::json attrs_json;
        for (const auto& [key, value] : span.attributes) {
            attrs_json[key] = AttributeValueToJson(value);
        }
        span_json["attributes"] = attrs_json;

        // Add resource attributes
        nlohmann::json res_attrs_json;
        for (const auto& [key, value] : span.resource_attributes) {
            res_attrs_json[key] = value;
        }
        span_json["resource_attributes"] = res_attrs_json;

        json_array.push_back(span_json);
    }

    return json_array.dump();
}
```

#### Async Export

```cpp
#ifdef PYFLARE_HAS_RDKAFKA
absl::Status KafkaExporter::Impl::Export(const std::vector<Span>& spans) {
    std::string payload = SerializeSpansToJson(spans);

    // Produce message
    int err = rd_kafka_produce(
        rd_kafka_topic_new(producer_, config_.topics.traces.c_str(), nullptr),
        RD_KAFKA_PARTITION_UA,              // Auto partition
        RD_KAFKA_MSG_F_COPY,                // Copy payload
        const_cast<char*>(payload.data()),
        payload.size(),
        nullptr, 0,                         // No key
        nullptr                             // No opaque
    );

    if (err != 0) {
        return absl::InternalError(
            absl::StrCat("Kafka produce failed: ", rd_kafka_err2str(rd_kafka_last_error())));
    }

    messages_sent_++;

    // Poll for delivery callbacks
    rd_kafka_poll(producer_, 0);

    return absl::OkStatus();
}

// Delivery callback
static void DeliveryCallback(rd_kafka_t* rk, const rd_kafka_message_t* msg, void* opaque) {
    if (msg->err) {
        spdlog::error("Kafka delivery failed: {}", rd_kafka_err2str(msg->err));
        // Increment failed counter
    } else {
        // Increment delivered counter
    }
}
#endif
```

### Configuration

```yaml
kafka:
  brokers:
    - "kafka-1:9092"
    - "kafka-2:9092"

  topics:
    traces: "pyflare.traces"
    metrics: "pyflare.metrics"
    logs: "pyflare.logs"

  producer:
    batch_size: 16384
    linger_ms: 5
    compression: "lz4"
    acks: "all"
    retries: 3
    enable_idempotence: true

  security:
    protocol: "SASL_SSL"
    sasl:
      mechanism: "SCRAM-SHA-256"
      username: "${KAFKA_USERNAME}"
      password: "${KAFKA_PASSWORD}"
```

---

## Python SDK

**Location:** `sdk/python/pyflare/`

### Module Structure

```
sdk/python/pyflare/
├── __init__.py          # Public API exports
├── sdk.py               # Core PyFlare class
├── types.py             # Type definitions
├── cost.py              # Cost calculator
└── integrations/
    ├── __init__.py
    ├── openai.py        # OpenAI auto-instrumentation
    └── anthropic.py     # Anthropic auto-instrumentation
```

### Core SDK (`sdk.py`)

```python
class PyFlare:
    """Core PyFlare SDK class."""

    _instance: ClassVar[Optional["PyFlare"]] = None

    def __init__(
        self,
        service_name: str,
        endpoint: str = "http://localhost:4317",
        environment: str = "development",
        version: str = "",
        enabled: bool = True,
        sample_rate: float = 1.0,
        debug: bool = False,
        use_http: bool = False,
        batch_export: bool = True,
        headers: Optional[dict[str, str]] = None,
        resource_attributes: Optional[dict[str, str]] = None,
    ):
        self.service_name = service_name
        self.endpoint = self._get_env("PYFLARE_ENDPOINT", endpoint)
        self.environment = self._get_env("PYFLARE_ENVIRONMENT", environment)
        self.version = version
        self.enabled = self._get_env_bool("PYFLARE_ENABLED", enabled)
        self.sample_rate = self._get_env_float("PYFLARE_SAMPLE_RATE", sample_rate)
        self.debug = debug

        if self.enabled:
            self._setup_tracer(use_http, batch_export, headers, resource_attributes)

        PyFlare._instance = self
```

### Trace Decorator

```python
def trace(
    name: Optional[str] = None,
    model_id: Optional[str] = None,
    inference_type: InferenceType = InferenceType.CUSTOM,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """Decorator for tracing functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            pyflare = PyFlare.get_instance()
            if not pyflare or not pyflare.enabled:
                return func(*args, **kwargs)

            with pyflare.tracer.start_as_current_span(span_name) as span:
                # Set attributes
                if model_id:
                    span.set_attribute("pyflare.model.id", model_id)
                span.set_attribute("pyflare.inference.type", inference_type.value)

                # Capture input
                if capture_input:
                    span.set_attribute("pyflare.input.preview",
                                      _truncate_preview(str(args) + str(kwargs)))

                try:
                    result = func(*args, **kwargs)

                    # Capture output
                    if capture_output:
                        span.set_attribute("pyflare.output.preview",
                                          _truncate_preview(str(result)))

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Handle async functions
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                # Similar implementation for async
                ...
            return async_wrapper

        return sync_wrapper
    return decorator
```

### Context Managers

```python
@contextmanager
def span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
         attributes: Optional[dict[str, Any]] = None) -> Iterator[Span]:
    """Create a span using context manager."""
    with self.tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span

@contextmanager
def llm_span(self, name: str, model_id: str, provider: str,
             **kwargs: Any) -> Iterator[Span]:
    """Create an LLM span with pre-configured attributes."""
    attributes = {
        "pyflare.model.id": model_id,
        "pyflare.model.provider": provider,
        "pyflare.inference.type": InferenceType.LLM.value,
        **kwargs,
    }
    with self.span(name, attributes=attributes) as span:
        yield span
```

---

## Cost Calculator

**Location:** `sdk/python/pyflare/cost.py`

### Pricing Database

```python
@dataclass
class ModelPricing:
    model_id: str
    provider: ModelProvider
    input_price_per_million: float   # USD per million tokens
    output_price_per_million: float  # USD per million tokens
    context_window: int = 0
    supports_vision: bool = False

# Built-in pricing database
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing("gpt-4o", ModelProvider.OPENAI, 2.50, 10.00, 128000),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", ModelProvider.OPENAI, 0.15, 0.60, 128000),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", ModelProvider.OPENAI, 10.00, 30.00, 128000),

    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing(
        "claude-3-5-sonnet-20241022", ModelProvider.ANTHROPIC, 3.00, 15.00, 200000),
    "claude-3-opus-20240229": ModelPricing(
        "claude-3-opus-20240229", ModelProvider.ANTHROPIC, 15.00, 75.00, 200000),

    # ... more models
}
```

### Cost Calculation

```python
class CostCalculator:
    def __init__(self):
        self.pricing = dict(MODEL_PRICING)

    def calculate(self, model_id: str, token_usage: TokenUsage) -> CostResult:
        pricing = self.get_pricing(model_id)
        if not pricing:
            return CostResult(
                input_micros=0, output_micros=0, total_micros=0,
                total_dollars=0.0, estimated=True
            )

        # Calculate cost in micro-dollars
        input_cost = (token_usage.input_tokens / 1_000_000) * pricing.input_price_per_million
        output_cost = (token_usage.output_tokens / 1_000_000) * pricing.output_price_per_million

        input_micros = int(input_cost * 1_000_000)
        output_micros = int(output_cost * 1_000_000)
        total_micros = input_micros + output_micros

        return CostResult(
            input_micros=input_micros,
            output_micros=output_micros,
            total_micros=total_micros,
            total_dollars=total_micros / 1_000_000,
            estimated=False,
        )

    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        # Exact match
        if model_id in self.pricing:
            return self.pricing[model_id]

        # Prefix match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
        for known_model, pricing in self.pricing.items():
            if model_id.startswith(known_model):
                return pricing

        return None
```

### Usage

```python
from pyflare.cost import calculate_cost, get_cost_calculator

# Simple usage
result = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
print(f"Cost: ${result.total_dollars:.4f}")

# Custom pricing
calculator = get_cost_calculator()
calculator.add_pricing(ModelPricing(
    model_id="my-custom-model",
    provider=ModelProvider.CUSTOM,
    input_price_per_million=1.0,
    output_price_per_million=2.0,
))
```

---

# Phase 2 Components

This section documents the Phase 2 components added to PyFlare.

## PSI Drift Detector

**Location:** `src/processor/drift/psi_detector.cpp`

The PSI (Population Stability Index) detector measures distribution shift for numerical and categorical features.

### Class Structure

```cpp
class PSIDriftDetector : public DriftDetector {
public:
    explicit PSIDriftDetector(PSIConfig config = {});

    absl::Status SetReference(const Distribution& reference) override;
    absl::StatusOr<DriftResult> Compute(
        const std::vector<DataPoint>& current_batch) override;

    DriftType Type() const override { return DriftType::kFeature; }
    std::string Name() const override { return "PSIDriftDetector"; }

    double ComputeFeaturePSI(const std::vector<double>& ref_values,
                             const std::vector<double>& cur_values) const;

private:
    void BuildBins(const std::vector<double>& values,
                   std::vector<double>& bin_edges,
                   std::vector<double>& bin_percentages) const;

    PSIConfig config_;
    Distribution reference_;
    std::vector<std::vector<double>> reference_bin_edges_;
    std::vector<std::vector<double>> reference_bin_percentages_;
};
```

### PSI Formula

```
PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
```

Interpretation:
- PSI < 0.1: No significant shift
- 0.1 ≤ PSI < 0.25: Moderate shift, investigate
- PSI ≥ 0.25: Significant shift, action required

### Configuration

```cpp
struct PSIConfig {
    double threshold = 0.2;
    size_t num_bins = 10;
    size_t min_samples_per_bin = 5;
    double epsilon = 1e-10;  // Avoid log(0)
};
```

---

## MMD Drift Detector

**Location:** `src/processor/drift/mmd_detector.cpp`

The MMD (Maximum Mean Discrepancy) detector measures embedding distribution drift using kernel methods.

### Class Structure

```cpp
class MMDDriftDetector : public DriftDetector {
public:
    explicit MMDDriftDetector(MMDConfig config = {});

    absl::Status SetReferenceEmbeddings(
        const std::vector<std::vector<float>>& embeddings);

    absl::StatusOr<DriftResult> ComputeFromEmbeddings(
        const std::vector<std::vector<float>>& embeddings);

    std::vector<float> GetReferenceCentroid() const;
    double ComputeCentroidDrift(
        const std::vector<std::vector<float>>& embeddings);

private:
    double RBFKernel(const std::vector<float>& x,
                     const std::vector<float>& y) const;
    double ComputeMMDSquared(const std::vector<std::vector<float>>& X,
                             const std::vector<std::vector<float>>& Y) const;
    double MedianHeuristic(const std::vector<std::vector<float>>& X) const;
    double PermutationTest(const std::vector<std::vector<float>>& X,
                           const std::vector<std::vector<float>>& Y,
                           double observed_mmd) const;

    MMDConfig config_;
    std::vector<std::vector<float>> reference_embeddings_;
    std::vector<float> reference_centroid_;
    double sigma_ = 1.0;
};
```

### MMD Formula

The unbiased MMD² estimator:

```
MMD² = 1/(m*(m-1)) × Σ_{i≠j} k(x_i, x_j)
     + 1/(n*(n-1)) × Σ_{i≠j} k(y_i, y_j)
     - 2/(m*n) × Σ_{i,j} k(x_i, y_j)
```

Where k is the RBF kernel: `k(x,y) = exp(-||x-y||² / (2σ²))`

### Configuration

```cpp
struct MMDConfig {
    double threshold = 0.1;
    double rbf_sigma = 0.0;  // 0 = auto via median heuristic
    size_t num_permutations = 100;
    double p_value_threshold = 0.05;
    size_t max_samples = 0;  // 0 = use all
    uint64_t random_seed = 0;  // 0 = random
};
```

---

## Budget Manager

**Location:** `src/processor/cost/budget_manager.cpp`

The Budget Manager tracks and enforces spend limits across multiple dimensions.

### Class Structure

```cpp
class BudgetManager {
public:
    BudgetManager(std::shared_ptr<storage::RedisClient> redis,
                  BudgetManagerConfig config = {});

    absl::Status Initialize();
    absl::Status Shutdown();

    // Budget CRUD
    absl::Status CreateBudget(const BudgetConfig& config);
    absl::StatusOr<BudgetConfig> GetBudget(BudgetDimension dimension,
                                            const std::string& dimension_value);
    absl::Status DeleteBudget(BudgetDimension dimension,
                              const std::string& dimension_value);

    // Budget operations
    absl::StatusOr<BudgetCheckResult> CheckBudget(
        BudgetDimension dimension,
        const std::string& dimension_value,
        int64_t proposed_spend_micros = 0);

    absl::Status RecordSpend(BudgetDimension dimension,
                              const std::string& dimension_value,
                              int64_t spend_micros);

    absl::StatusOr<BudgetStatus> GetStatus(BudgetDimension dimension,
                                            const std::string& dimension_value);

    // Alerts
    void RegisterAlertCallback(BudgetAlertCallback callback);

    // Forecasting
    absl::StatusOr<int64_t> ForecastSpend(BudgetDimension dimension,
                                           const std::string& dimension_value);

private:
    std::shared_ptr<storage::RedisClient> redis_;
    BudgetManagerConfig config_;
    std::vector<BudgetAlertCallback> alert_callbacks_;
    std::unordered_map<std::string, LocalCounter> local_counters_;
};
```

### Budget Dimensions

```cpp
enum class BudgetDimension {
    kGlobal,      // Total spend
    kUser,        // Per-user budget
    kModel,       // Per-model budget
    kFeature,     // Per-feature/endpoint
    kTeam,        // Per-team budget
    kEnvironment  // Per-environment
};
```

### Budget Periods

```cpp
enum class BudgetPeriod {
    kHourly,   // 3600 seconds
    kDaily,    // 86400 seconds
    kWeekly,   // 604800 seconds
    kMonthly   // 2592000 seconds (30 days)
};
```

---

## LLM Judge Evaluator

**Location:** `src/processor/eval/llm_judge.cpp`

The LLM Judge uses an LLM to evaluate outputs for hallucination and quality.

### Class Structure

```cpp
class LLMJudgeEvaluator : public Evaluator {
public:
    explicit LLMJudgeEvaluator(LLMJudgeConfig config = {});

    absl::Status Initialize();
    absl::StatusOr<EvalResult> Evaluate(const InferenceRecord& record) override;
    absl::StatusOr<std::vector<EvalResult>> EvaluateBatch(
        const std::vector<InferenceRecord>& records) override;

    // Specialized evaluations
    absl::StatusOr<JudgeVerdict> EvaluateHallucination(
        const InferenceRecord& record);
    absl::StatusOr<JudgeVerdict> EvaluateAgainstReference(
        const InferenceRecord& record);
    absl::StatusOr<JudgeVerdict> ComparePairwise(
        const std::string& input,
        const std::string& output_a,
        const std::string& output_b,
        const std::string& criteria = "");

private:
    absl::StatusOr<std::string> CallLLM(
        const std::string& system_prompt,
        const std::string& user_prompt);
    JudgeVerdict ParseVerdict(const std::string& response);
    std::string BuildHallucinationPrompt(const InferenceRecord& record);

    LLMJudgeConfig config_;
    JudgePromptTemplate hallucination_prompt_;
};
```

### Judge Verdict

```cpp
struct JudgeVerdict {
    enum class Result { kPass, kFail, kUnsure, kError };

    Result result = Result::kUnsure;
    double score = 0.0;
    std::string explanation;
    bool has_hallucination = false;
    bool has_factual_error = false;
    bool has_contradiction = false;
    bool has_unsupported_claim = false;
};
```

### Hallucination Detection Prompt

The default prompt template:

```
System: You are a hallucination detector. Analyze whether the output contains
claims not supported by the provided context.

User:
Context: {context}
Question: {input}
Answer: {output}

Evaluate if the answer contains hallucinations (claims not in context).
Return JSON: {"has_hallucination": bool, "score": 0-1, "explanation": "..."}
```

---

## RAG Evaluator

**Location:** `src/processor/eval/rag_evaluator.cpp`

The RAG Evaluator measures retrieval-augmented generation quality.

### Class Structure

```cpp
class RAGEvaluator : public Evaluator {
public:
    explicit RAGEvaluator(RAGEvaluatorConfig config);

    absl::StatusOr<RAGMetrics> EvaluateRAG(const InferenceRecord& record);
    absl::StatusOr<double> EvaluateContextRelevance(
        const std::string& query,
        const std::vector<std::string>& contexts);
    absl::StatusOr<double> EvaluateFaithfulness(
        const std::string& answer,
        const std::vector<std::string>& contexts);
    absl::StatusOr<double> EvaluateAnswerRelevance(
        const std::string& query,
        const std::string& answer);
    absl::StatusOr<double> EvaluateGroundedness(
        const std::string& answer,
        const std::vector<std::string>& contexts);

private:
    std::vector<std::string> ExtractClaims(const std::string& text);
    bool IsClaimSupported(const std::string& claim,
                          const std::vector<std::string>& contexts);
    double CalculateKeywordOverlap(const std::string& text1,
                                   const std::string& text2);
    double CalculateOverallScore(const RAGMetrics& metrics) const;

    RAGEvaluatorConfig config_;
};
```

### RAG Metrics

```cpp
struct RAGMetrics {
    double context_relevance = 0.0;    // Context relevance to query
    double faithfulness = 0.0;         // Answer faithful to context
    double answer_relevance = 0.0;     // Answer addresses query
    double groundedness = 0.0;         // Claims grounded in context
    bool has_hallucination = false;
    bool has_irrelevant_context = false;
    double overall_score = 0.0;
    std::vector<std::string> issues;
};
```

---

## Slice Analyzer

**Location:** `src/processor/rca/slice_analyzer.cpp`

The Slice Analyzer identifies problematic data segments.

### Class Structure

```cpp
class SliceAnalyzer {
public:
    SliceAnalyzer(std::shared_ptr<storage::ClickHouseClient> clickhouse,
                  SliceAnalyzerConfig config = {});

    absl::Status Initialize();

    absl::StatusOr<std::vector<SliceAnalysisResult>> AnalyzeSlices(
        const std::string& model_id,
        SliceMetric metric);

    absl::StatusOr<std::vector<SliceAnalysisResult>> AnalyzeDimension(
        const std::string& model_id,
        SliceDimension dimension,
        SliceMetric metric);

    absl::StatusOr<std::vector<SliceAnalysisResult>> FindTopProblematicSlices(
        const std::string& model_id = "",
        size_t limit = 10);

    absl::StatusOr<double> GetBaseline(const std::string& model_id,
                                        SliceMetric metric);

private:
    std::string BuildSliceQuery(const std::string& model_id,
                                 SliceDimension dimension,
                                 SliceMetric metric);
    void CalculateSignificance(SliceAnalysisResult& result,
                                size_t total_samples);
    void CalculateImpact(SliceAnalysisResult& result);

    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    SliceAnalyzerConfig config_;
    std::unordered_map<std::string, double> baselines_;
};
```

### Slice Dimensions

```cpp
enum class SliceDimension {
    kModel, kUser, kFeature, kInputLength, kOutputLength,
    kLatency, kTimeOfDay, kDayOfWeek, kPromptTemplate,
    kProvider, kEnvironment, kCustom
};
```

### Slice Metrics

```cpp
enum class SliceMetric {
    kErrorRate, kLatencyP50, kLatencyP95, kLatencyP99,
    kCost, kTokenUsage, kToxicityRate, kHallucinationRate,
    kDriftScore, kCustom
};
```

### Analysis Result

```cpp
struct SliceAnalysisResult {
    SliceDefinition definition;
    SliceMetric metric;

    size_t sample_count = 0;
    double metric_value = 0.0;
    double baseline_value = 0.0;
    double deviation = 0.0;
    double deviation_percentage = 0.0;

    double p_value = 0.0;
    bool is_statistically_significant = false;
    double impact_score = 0.0;
};
```

---

## Pattern Detector

**Location:** `src/processor/rca/pattern_detector.cpp`

The Pattern Detector identifies failure patterns and anomalies.

### Class Structure

```cpp
class PatternDetector {
public:
    PatternDetector(std::shared_ptr<storage::ClickHouseClient> clickhouse,
                    PatternDetectorConfig config = {});

    absl::StatusOr<std::vector<Pattern>> DetectPatterns(
        const std::string& model_id = "");

    absl::StatusOr<std::vector<Pattern>> DetectErrorSpikes(
        const std::string& model_id);
    absl::StatusOr<std::vector<Pattern>> DetectLatencyDegradation(
        const std::string& model_id);
    absl::StatusOr<std::vector<Pattern>> DetectQualityDrops(
        const std::string& model_id);
    absl::StatusOr<std::vector<Pattern>> DetectCostAnomalies(
        const std::string& model_id);

    std::vector<std::string> GenerateSuggestedActions(const Pattern& pattern);

private:
    std::shared_ptr<storage::ClickHouseClient> clickhouse_;
    PatternDetectorConfig config_;
};
```

### Pattern Types

```cpp
enum class PatternType {
    kErrorSpike,           // Sudden increase in errors
    kLatencyDegradation,   // Gradual latency increase
    kQualityDrop,          // Evaluation score decrease
    kCostAnomaly,          // Unusual spending pattern
    kDriftCorrelation,     // Drift correlates with issues
    kTemporalPattern,      // Time-based patterns
    kUserSegmentIssue,     // Issues in user segment
    kCustom
};
```

---

## Failure Clusterer

**Location:** `src/processor/rca/failure_cluster.cpp`

The Failure Clusterer groups similar failures for efficient debugging.

### Class Structure

```cpp
class FailureClusterer {
public:
    explicit FailureClusterer(FailureClustererConfig config = {});

    absl::StatusOr<std::vector<FailureCluster>> ClusterFailures(
        const std::vector<FailureRecord>& failures);

    std::string FindRepresentativeError(const FailureCluster& cluster);
    std::vector<std::string> ExtractCommonKeywords(
        const FailureCluster& cluster);
    double CalculateTextSimilarity(const std::string& text1,
                                   const std::string& text2);

private:
    double JaccardSimilarity(const std::set<std::string>& a,
                             const std::set<std::string>& b);
    std::set<std::string> Tokenize(const std::string& text);

    FailureClustererConfig config_;
};
```

### Failure Cluster

```cpp
struct FailureCluster {
    std::string id;
    std::string name;
    size_t size = 0;
    std::string representative_error;
    std::vector<std::string> common_keywords;
    double severity = 0.0;
    std::vector<std::string> member_ids;
    std::vector<std::string> affected_traces;
};
```

---

## Query API Handlers

**Location:** `src/query/handlers/`

REST API handlers for Phase 2 data access.

### Handler Base Class

```cpp
class Handler {
public:
    virtual ~Handler() = default;

    virtual HttpResponse Handle(const HttpRequest& request,
                                 const HandlerContext& context) = 0;

    virtual std::string GetRoute() const = 0;
    virtual std::vector<HttpMethod> GetMethods() const = 0;
};
```

### Available Handlers

| Route | Handler | Description |
|-------|---------|-------------|
| `/api/v1/traces` | `ListTracesHandler` | List and filter traces |
| `/api/v1/traces/:id` | `GetTraceHandler` | Get trace details |
| `/api/v1/drift/alerts` | `DriftAlertsHandler` | Get drift alerts |
| `/api/v1/drift/heatmap` | `DriftHeatmapHandler` | Get drift heatmap data |
| `/api/v1/costs/summary` | `CostSummaryHandler` | Cost aggregations |
| `/api/v1/costs/budgets` | `BudgetsHandler` | Budget management |
| `/api/v1/evaluations` | `ListEvaluationsHandler` | Evaluation results |
| `/api/v1/rca/patterns` | `RCAPatternsHandler` | Detected patterns |
| `/api/v1/rca/slices` | `RCASlicesHandler` | Problematic slices |
| `/api/v1/rca/clusters` | `RCAClustersHandler` | Failure clusters |

---

## Next Steps

- [Extension Guide](./extending.md) - How to add new features
- [Building](./building.md) - Build instructions
