# PyFlare Grafana Integration

Use PyFlare data in Grafana dashboards for unified observability.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Query Types](#query-types)
- [Building Dashboards](#building-dashboards)
- [Alerting](#alerting)
- [Pre-built Dashboards](#pre-built-dashboards)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Grafana 10.0 or higher
- PyFlare server running and accessible

### Plugin Installation

#### From Grafana Catalog

1. Open Grafana
2. Navigate to **Configuration** > **Plugins**
3. Search for "PyFlare"
4. Click **Install**

#### Manual Installation

1. Download the plugin:
   ```bash
   grafana-cli plugins install pyflare-datasource
   ```

2. Or build from source:
   ```bash
   cd grafana/pyflare-datasource
   npm install
   npm run build
   ```

3. Copy to Grafana plugins directory:
   ```bash
   cp -r dist /var/lib/grafana/plugins/pyflare-datasource
   ```

4. Restart Grafana:
   ```bash
   systemctl restart grafana-server
   ```

---

## Configuration

### Adding the Data Source

1. Go to **Configuration** > **Data Sources**
2. Click **Add data source**
3. Search for "PyFlare" and select it

### Connection Settings

| Setting | Description | Example |
|---------|-------------|---------|
| **URL** | PyFlare API endpoint | `http://localhost:8080` |
| **API Key** | Authentication key (optional) | `pf_key_xxx...` |
| **Timeout** | Request timeout in seconds | `30` |

### Default Filters

Configure default values for queries:

| Setting | Description | Example |
|---------|-------------|---------|
| **Default Service** | Pre-fill service filter | `chat-service` |
| **Default Model** | Pre-fill model filter | `gpt-4` |

### Testing Connection

Click **Save & Test** to verify the connection. You should see "Successfully connected to PyFlare".

---

## Query Types

The PyFlare data source supports five query types:

### Metrics Query

Query time-series metrics data.

**Options:**
- **Metric**: Select from available metrics
  - `request_count` - Number of requests
  - `latency_ms` - Latency in milliseconds
  - `tokens_input` - Input token count
  - `tokens_output` - Output token count
  - `tokens_total` - Total tokens
  - `cost_micros` - Cost in micro-dollars
  - `error_rate` - Error percentage
  - `drift_score` - Drift detection score
  - `eval_score` - Evaluation score
  - `toxicity_score` - Toxicity score

- **Aggregation**: How to aggregate values
  - `avg`, `sum`, `min`, `max`, `count`
  - `p50`, `p95`, `p99` (percentiles)

- **Group By**: Split by dimensions
  - `model_id`, `service_name`, `user_id`, `status`

**Example**: Average latency by model:
```
Metric: latency_ms
Aggregation: avg
Group By: model_id
```

### Traces Query

Query raw trace data for tables.

**Options:**
- **Status**: Filter by `ok`, `error`, or `all`
- **Min Duration**: Minimum latency in ms
- **Max Duration**: Maximum latency in ms
- **Service**: Filter by service name
- **Model**: Filter by model ID
- **Limit**: Maximum rows to return

**Fields returned:**
- Time, Trace ID, Service, Model
- Status, Duration, Input Tokens, Output Tokens, Cost

### Drift Query

Query drift detection data.

**Options:**
- **Drift Type**: Filter by drift type
  - `all` - All drift types
  - `feature` - Feature drift (PSI)
  - `embedding` - Embedding drift (MMD)
  - `concept` - Concept drift
  - `prediction` - Prediction drift

- **Threshold**: Optional drift threshold filter

**Fields returned:**
- Time, Feature Drift, Embedding Drift, Concept Drift, Prediction Drift

### Costs Query

Query cost analytics data.

**Options:**
- **Group By**: Dimension to group costs
  - `model` - By model
  - `service` - By service
  - `user` - By user
  - `feature` - By feature

**Fields returned:**
- Time, Cost, Tokens, Requests
- Group dimension (if specified)

### Evaluations Query

Query evaluation scores over time.

**Options:**
- **Metric**: Evaluation metric
  - `accuracy` - Accuracy score
  - `latency` - Response latency
  - `toxicity` - Toxicity score
  - `hallucination` - Hallucination rate
  - `relevance` - Relevance score

- **Model**: Filter by model ID

**Fields returned:**
- Time, Score, Count

---

## Building Dashboards

### Time Series Panel (Latency)

1. Add new panel
2. Select PyFlare data source
3. Configure query:
   - Query Type: `Metrics`
   - Metric: `latency_ms`
   - Aggregation: `p95`
   - Group By: `model_id`

4. Panel settings:
   - Visualization: Time series
   - Unit: milliseconds (ms)

### Stat Panel (Request Count)

1. Add new panel
2. Configure query:
   - Query Type: `Metrics`
   - Metric: `request_count`
   - Aggregation: `sum`

3. Panel settings:
   - Visualization: Stat
   - Graph mode: None
   - Color mode: Value

### Table Panel (Recent Traces)

1. Add new panel
2. Configure query:
   - Query Type: `Traces`
   - Status: `all`
   - Limit: `20`

3. Panel settings:
   - Visualization: Table
   - Column overrides for Status (value mapping)

### Drift Monitoring Panel

1. Add new panel
2. Configure query:
   - Query Type: `Drift`
   - Drift Type: `all`

3. Panel settings:
   - Visualization: Time series
   - Add threshold at 30% (warning)
   - Add threshold at 50% (critical)

### Cost Breakdown (Pie Chart)

1. Add new panel
2. Configure query:
   - Query Type: `Costs`
   - Group By: `model`

3. Panel settings:
   - Visualization: Pie chart
   - Value field: Cost
   - Label field: model

---

## Alerting

Create Grafana alerts based on PyFlare data.

### High Latency Alert

1. Edit your latency panel
2. Go to **Alert** tab
3. Click **Create alert rule**
4. Configure:
   - Name: `High Latency Alert`
   - Evaluate: Every `1m` for `5m`
   - Condition: When `avg()` of query `A` is above `2000`
5. Add notification channel

### Drift Detection Alert

1. Create drift panel query
2. Add alert rule:
   - Condition: When `max()` of `Feature Drift` is above `0.3`
   - Message: `Drift detected: {{ $value }}`

### Cost Budget Alert

1. Create cost panel
2. Add alert rule:
   - Condition: When `sum()` of `Cost` is above `1000`
   - Frequency: Daily
   - Message: `Daily cost exceeded $1000`

### Error Rate Alert

1. Create error rate panel:
   - Metric: `error_rate`
   - Aggregation: `avg`
2. Add alert:
   - Condition: When above `5%`
   - Duration: `5m`

---

## Pre-built Dashboards

Import ready-to-use dashboards.

### ML Overview Dashboard

**ID**: `pyflare-ml-overview`

Includes:
- Request volume over time
- Latency percentiles (p50, p95, p99)
- Error rate
- Token usage by model
- Cost summary
- Top models by usage

### Drift Monitoring Dashboard

**ID**: `pyflare-drift-monitoring`

Includes:
- Drift score timeline
- Feature drift heatmap
- Embedding drift
- Drift alerts
- Model comparison

### Cost Analytics Dashboard

**ID**: `pyflare-cost-analytics`

Includes:
- Cost over time
- Cost by model (pie)
- Cost by service (bar)
- Token efficiency
- Budget tracking
- Forecast projection

### Import Instructions

1. Go to **Dashboards** > **Import**
2. Enter dashboard ID or upload JSON
3. Select PyFlare data source
4. Click **Import**

---

## Variables and Templates

### Model Variable

Create a variable for model selection:

1. Go to dashboard settings > Variables
2. Click **Add variable**
3. Configure:
   - Name: `model`
   - Type: Query
   - Data source: PyFlare
   - Query: `SELECT DISTINCT model_id FROM traces`
   - Multi-value: Yes

4. Use in queries: `Model: $model`

### Service Variable

```
Name: service
Query: SELECT DISTINCT service_name FROM traces
```

### Time Range Templates

Use Grafana's built-in time range with PyFlare queries:

- `$__from` - Start time
- `$__to` - End time
- `$__interval` - Auto-calculated interval

---

## Troubleshooting

### No Data Returned

1. **Check connection**: Test data source in settings
2. **Verify time range**: Ensure data exists in selected range
3. **Check filters**: Remove filters to test
4. **View query**: Use Query Inspector to see raw query

### Connection Errors

1. **Network**: Verify Grafana can reach PyFlare server
2. **Firewall**: Check port 8080 is open
3. **CORS**: Ensure PyFlare allows Grafana origin
4. **Authentication**: Verify API key is valid

### Slow Queries

1. **Reduce time range**: Query smaller periods
2. **Add filters**: Narrow down data
3. **Increase timeout**: Adjust in data source settings
4. **Check server**: Monitor PyFlare server resources

### Missing Metrics

1. Verify data is being collected
2. Check metric name spelling
3. Ensure model/service exists
4. Check SDK is exporting correctly

### Dashboard Permissions

1. Verify user has dashboard edit access
2. Check data source permissions
3. Review organization settings

---

## API Reference

### Health Check

```
GET /api/v1/health
```

Returns:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Metrics Query

```
GET /api/v1/metrics/query?metric=latency_ms&aggregation=avg&group_by=model_id&start_time=...&end_time=...
```

### Traces Query

```
GET /api/v1/traces?service_name=...&model_id=...&status=...&limit=100
```

### Drift Query

```
GET /api/v1/drift/timeline?model_id=...&drift_type=all&start_time=...&end_time=...
```

### Costs Query

```
GET /api/v1/costs/timeline?group_by=model&start_time=...&end_time=...
```

---

## Resources

- [PyFlare Documentation](https://pyflare.io/docs)
- [Grafana Documentation](https://grafana.com/docs)
- [Example Dashboards](https://github.com/pyflare/pyflare/tree/main/grafana/dashboards)
- [Community Forum](https://community.grafana.com)
