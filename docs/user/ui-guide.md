# PyFlare UI Guide

Complete guide to using the PyFlare web dashboard for ML observability.

## Table of Contents

- [Getting Started](#getting-started)
- [Dashboard Overview](#dashboard-overview)
- [Trace Explorer](#trace-explorer)
- [Drift Dashboard](#drift-dashboard)
- [Cost Analytics](#cost-analytics)
- [Evaluations](#evaluations)
- [Settings & Configuration](#settings--configuration)

---

## Getting Started

### Accessing the Dashboard

The PyFlare dashboard is available at `http://localhost:3000` by default when running PyFlare locally.

### Authentication

PyFlare supports two authentication methods:

1. **Email/Password**: For team members with full dashboard access
2. **API Keys**: For programmatic access and integrations

To log in:
1. Navigate to the login page
2. Enter your credentials or API key
3. Click "Sign In"

### Dark/Light Theme

Toggle between dark and light themes using the theme button in the top navigation bar. The setting persists across sessions.

---

## Dashboard Overview

The main dashboard provides a high-level view of your ML operations:

### Key Metrics Cards

- **Total Requests**: Number of inference requests in the selected time period
- **Average Latency**: Mean response time across all requests
- **Error Rate**: Percentage of failed requests
- **Total Cost**: Cumulative cost for the time period

### Quick Navigation

The sidebar provides access to all major sections:
- **Dashboard**: Overview and key metrics
- **Traces**: Detailed trace exploration
- **Drift**: Drift detection and monitoring
- **Costs**: Cost analytics and budgeting
- **Evaluations**: Quality evaluation results
- **Settings**: System configuration

---

## Trace Explorer

The Trace Explorer allows you to search, filter, and analyze individual inference traces.

### List View

The default list view shows all traces with:
- **Trace ID**: Unique identifier (clickable)
- **Service**: Source service name
- **Model**: Model used for inference
- **Status**: Success (green) or Error (red)
- **Duration**: Request latency in milliseconds
- **Tokens**: Input/output token counts
- **Cost**: Estimated cost in USD
- **Timestamp**: When the request occurred

### Search Query Language

Use the search bar with query syntax:

```
service:chat-api model:gpt-4 status:error duration:>2000 time:last-24h
```

**Supported filters:**
- `service:<name>` - Filter by service name
- `model:<id>` - Filter by model ID
- `status:ok|error` - Filter by status
- `duration:>N` or `duration:<N` - Filter by duration (ms)
- `time:last-1h|last-24h|last-7d` - Time range filter
- `has:drift` - Traces with drift detected
- `has:safety` - Traces with safety issues

### Filter Sidebar

Toggle the filter sidebar to apply faceted filters:

1. **Time Range**: Quick presets and custom range
2. **Services**: Multi-select available services
3. **Models**: Filter by specific models
4. **Status**: Success, Error, or All
5. **Duration**: Slider for min/max duration
6. **Tags**: Filter by custom tags

### Trace Detail View

Click any trace to view:

1. **Summary Panel**
   - Request metadata
   - Token usage breakdown
   - Cost calculation
   - Evaluation scores

2. **Span Waterfall**
   - Visual timeline of all spans
   - Critical path highlighting
   - Nested span hierarchy
   - Click spans for details

3. **Input/Output**
   - Full prompt text
   - Complete response
   - System message (if any)

4. **Attributes**
   - Custom span attributes
   - Model parameters
   - User/session info

### Trace Comparison

Compare two traces side-by-side:

1. Select two traces using checkboxes
2. Click "Compare" button
3. View differences in:
   - Latency and tokens
   - Input/output content
   - Span structure

### Live Stream

View traces in real-time:

1. Switch to "Live Stream" view
2. Traces appear as they're received
3. Click any trace to pause and view details
4. Filter by status or model while streaming

### Export

Export trace data in various formats:

1. Click "Export" button
2. Choose format (JSON, CSV)
3. Select fields to include
4. Download file

---

## Drift Dashboard

Monitor data distribution drift across your models.

### Overview Tab

The overview shows drift status across four dimensions:

1. **Feature Drift**
   - Score: PSI-based measurement
   - Status: Healthy (green), Warning (yellow), Critical (red)
   - Trend indicator

2. **Embedding Drift**
   - Score: MMD-based measurement
   - Tracks semantic shifts in embeddings

3. **Concept Drift**
   - Changes in label/output relationships
   - Important for classification models

4. **Prediction Drift**
   - Output distribution changes
   - Indicates model behavior shifts

### Timeline Chart

The timeline visualizes drift scores over time:

- **X-axis**: Time (adjustable granularity)
- **Y-axis**: Drift score (0-100%)
- **Threshold line**: Configurable alert threshold
- **Legend**: Toggle drift types on/off

Hover over points to see:
- Exact timestamp
- Individual drift scores
- Alert indicators

### Feature Analysis Tab

Detailed per-feature drift analysis:

1. **Feature Table**
   - Feature name and type
   - Drift score with status badge
   - P-value for statistical significance
   - Trend indicator
   - Feature importance

2. **Sorting**: Click column headers to sort
3. **Filtering**: Toggle between All, Drifted, Stable

4. **Distribution View**: Click any feature to expand:
   - Reference distribution (gray)
   - Current distribution (colored)
   - Overlay comparison

### Alert Settings Tab

Configure drift monitoring alerts:

1. **Enable/Disable**: Toggle monitoring on/off

2. **Thresholds**: Set per-drift-type thresholds
   - Feature drift threshold
   - Embedding drift threshold
   - Concept drift threshold
   - Prediction drift threshold

3. **Evaluation Window**: How often to check
   - 1 hour, 6 hours, 24 hours

4. **Notifications**: Select channels
   - Slack integration
   - Email alerts
   - PagerDuty integration

5. **Severity Filter**: Minimum severity to alert
   - Low, Medium, High, Critical

### Model Selector

Use the dropdown to switch between models. Each model has independent:
- Drift scores
- Thresholds
- Alert configuration

---

## Cost Analytics

Track, analyze, and control your AI/ML spending.

### Overview Tab

Key cost metrics at a glance:

1. **Total Cost**: Sum of all costs in period
2. **Total Tokens**: Token consumption
3. **Avg Cost/Request**: Cost efficiency metric
4. **Total Requests**: Request volume

Each metric shows:
- Current value
- Change percentage vs previous period
- Trend direction

**Budget Progress Bar**: Visual indicator of budget consumption with alert threshold.

### Cost Breakdown Tab

Detailed cost analysis by dimension:

1. **Group By**: Select grouping dimension
   - Model
   - Service
   - User
   - Feature

2. **Breakdown Table**:
   - Dimension value
   - Total cost
   - Token count
   - Request count
   - Average cost per request
   - Percentage of total (visual bar)

3. **Expandable Rows**: Click to see nested breakdown

4. **Export**: Download breakdown as JSON/CSV

### Token Usage Chart

Visualize token consumption:

- **Stacked Area Chart**: Input vs output tokens
- **Cache Hits**: Optional overlay for cached responses
- **Time Granularity**: Adjust for hourly/daily/weekly view

### Budgets Tab

Track multiple budget configurations:

1. **Summary Cards**:
   - Total budget across all configs
   - Total used
   - Budgets at risk (near threshold)
   - Budgets exceeded

2. **Budget Cards**: Each budget shows:
   - Name and scope
   - Used / Limit amounts
   - Progress bar with status
   - Reset date
   - Alert indicators

3. **Budget History Chart**: Track spending over time

### Forecast Tab

Predictive cost analysis:

1. **Projections**:
   - Projected end-of-period cost
   - Projected change vs previous
   - Confidence level
   - Budget impact assessment

2. **Forecast Chart**:
   - Historical data (solid line)
   - Forecast data (dashed line)
   - Confidence interval (shaded area)
   - Budget limit reference line

3. **Toggle**: Show/hide confidence interval

### Budget Settings Tab

Configure cost budgets:

1. **Add Budget**: Create new budget with:
   - Name
   - Limit amount
   - Period (daily/weekly/monthly)
   - Alert threshold percentage
   - Scope (global, model, service, user)

2. **Edit Existing**: Modify budget parameters

3. **Delete**: Remove budgets

4. **Preview**: See configuration summary before saving

---

## Evaluations

Quality monitoring for LLM outputs.

### Overview

View aggregate evaluation scores:

- **Accuracy Score**: Average correctness
- **Toxicity Score**: Safety metric
- **Hallucination Rate**: Factuality issues
- **Relevance Score**: Response quality

### Evaluation Timeline

Track evaluation metrics over time:

- Select metric to display
- View trends and anomalies
- Identify degradation patterns

### Failed Evaluations

List of traces that failed evaluation checks:

- Trace ID and timestamp
- Failed evaluation type
- Score and threshold
- Quick link to trace detail

---

## Settings & Configuration

### Profile Settings

- Update display name
- Change password
- Enable 2FA

### API Keys

Manage API keys for programmatic access:

1. **Create Key**: Generate new key with:
   - Name/description
   - Expiration date
   - Permission scope

2. **View Keys**: List all active keys
3. **Revoke**: Disable compromised keys

### Notifications

Configure notification preferences:

- Email alerts
- Slack integration
- PagerDuty webhooks
- Custom webhook URLs

### Team Management

For organization accounts:

- Invite team members
- Assign roles (Admin, Editor, Viewer)
- Remove members

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + K` | Open search |
| `Ctrl/Cmd + /` | Show shortcuts |
| `G then D` | Go to Dashboard |
| `G then T` | Go to Traces |
| `G then R` | Go to Drift |
| `G then C` | Go to Costs |
| `?` | Open help |

---

## Troubleshooting

### No Data Showing

1. Verify PyFlare SDK is configured correctly
2. Check collector endpoint is reachable
3. Ensure traces are being exported
4. Check time range filter

### Slow Performance

1. Reduce time range
2. Add filters to narrow results
3. Check browser console for errors

### Authentication Issues

1. Clear browser cache and cookies
2. Verify API key is valid and not expired
3. Check network connectivity

### Export Failures

1. Reduce export size with filters
2. Try different export format
3. Check browser download permissions

---

## Getting Help

- **Documentation**: https://pyflare.io/docs
- **GitHub Issues**: https://github.com/pyflare/pyflare/issues
- **Discord Community**: https://discord.gg/pyflare
