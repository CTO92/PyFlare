# PyFlare Phase 4 Development Plan
# UI & Polish - Detailed Implementation Guide

> **Document Version**: 1.0
> **Status**: Planning
> **Created**: 2026-01-15
> **Estimated Duration**: 4 weeks

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Stage 1: UI Foundation Enhancement](#3-stage-1-ui-foundation-enhancement)
4. [Stage 2: Trace Explorer](#4-stage-2-trace-explorer)
5. [Stage 3: Drift Dashboard](#5-stage-3-drift-dashboard)
6. [Stage 4: Cost Analytics](#6-stage-4-cost-analytics)
7. [Stage 5: SDK Integration Enhancement](#7-stage-5-sdk-integration-enhancement)
8. [Stage 6: Grafana Plugin](#8-stage-6-grafana-plugin)
9. [Stage 7: Documentation](#9-stage-7-documentation)
10. [Stage 8: Testing & Performance](#10-stage-8-testing--performance)
11. [Implementation Order & Dependencies](#11-implementation-order--dependencies)
12. [Verification Checklist](#12-verification-checklist)

---

## 1. Executive Summary

### Objective
Complete the PyFlare platform with a production-ready web UI, enhanced SDK integrations, Grafana plugin, comprehensive documentation, and thorough testing.

### Key Deliverables
- **Complete Web UI**: Full-featured trace explorer, drift dashboards, cost analytics
- **Enhanced SDK Integrations**: Production-ready LangChain, OpenAI, PyTorch, PyFlame integrations with documentation
- **Grafana Plugin**: Data source and panel plugins for existing Grafana deployments
- **Documentation**: API reference, getting started guide, deployment guide, architecture docs
- **Testing**: E2E test suite, performance benchmarks, security audit

### Existing Assets (from Phase 1-3)
- UI foundation with Vite, React 18, TypeScript, Tailwind CSS
- Basic pages: Dashboard, Traces, TraceDetail, Drift, Costs, Alerts, Settings
- Components: TraceViewer, DriftHeatmap, CostCharts, RCAExplorer, EvaluationResults, IntelligenceDashboard, AlertsPanel
- Hooks: useTraces, useDrift, useCosts
- API client and services
- SDK integrations (basic): openai.py, langchain.py, pytorch.py, pyflame.py, anthropic.py

---

## 2. Current State Analysis

### 2.1 UI Components Status

| Component | File | Status | Phase 4 Work Needed |
|-----------|------|--------|---------------------|
| Layout | `Layout.tsx` | Exists | Add breadcrumbs, user menu, theme toggle |
| Dashboard | `Dashboard.tsx` | Exists | Add real-time metrics, quick actions |
| Traces | `Traces.tsx` | Exists | Add advanced filters, bulk actions |
| TraceDetail | `TraceDetail.tsx` | Exists | Add span waterfall, comparison |
| Drift | `Drift.tsx` | Exists | Enhance with feature-level breakdown |
| Costs | `Costs.tsx` | Exists | Add budget alerts, forecasting |
| Alerts | `Alerts.tsx` | Exists | Integrate AlertsPanel fully |
| Settings | `Settings.tsx` | Exists | Add API key management, preferences |
| TraceViewer | `TraceViewer.tsx` | Exists | Add flame graph, JSON view |
| DriftHeatmap | `DriftHeatmap.tsx` | Exists | Add interactive drill-down |
| CostCharts | `CostCharts.tsx` | Exists | Add comparison, export |
| RCAExplorer | `RCAExplorer.tsx` | Exists | Add visualization graph |
| EvaluationResults | `EvaluationResults.tsx` | Exists | Add batch comparison |
| IntelligenceDashboard | `IntelligenceDashboard.tsx` | Exists | Add real-time updates |
| AlertsPanel | `AlertsPanel.tsx` | Exists | Add notification preferences |

### 2.2 SDK Integration Status

| Integration | File | Status | Phase 4 Work Needed |
|-------------|------|--------|---------------------|
| OpenAI | `openai.py` | Basic | Add streaming support, function calls |
| LangChain | `langchain.py` | Basic | Add agent tracing, chain visualization |
| PyTorch | `pytorch.py` | Basic | Add model profiling, gradient tracking |
| PyFlame | `pyflame.py` | Basic | Add native metric integration |
| Anthropic | `anthropic.py` | Basic | Add streaming, tool use tracing |

### 2.3 Missing Components

**New UI Components Needed:**
- `SpanWaterfall.tsx` - Waterfall visualization for spans
- `TraceComparison.tsx` - Side-by-side trace comparison
- `TraceSearch.tsx` - Advanced search with saved queries
- `RealTimeTraceStream.tsx` - WebSocket-based live traces
- `BudgetTracker.tsx` - Budget visualization and alerts
- `CostForecast.tsx` - Cost prediction charts
- `FeatureDriftBreakdown.tsx` - Feature-level drift analysis
- `DriftTimeline.tsx` - Historical drift visualization
- `ExportDialog.tsx` - Data export functionality
- `ThemeProvider.tsx` - Dark/light mode support
- `AuthProvider.tsx` - Authentication context
- `NotificationCenter.tsx` - In-app notifications

**New Hooks Needed:**
- `useWebSocket.ts` - WebSocket connection management
- `useAuth.ts` - Authentication state
- `useAlerts.ts` - Alert state and actions
- `useRCA.ts` - RCA operations
- `useIntelligence.ts` - Intelligence pipeline data
- `useNotifications.ts` - Notification management
- `useExport.ts` - Export functionality

---

## 3. Stage 1: UI Foundation Enhancement

### 3.1 Authentication System

**Files to create:**
- `ui/src/contexts/AuthContext.tsx`
- `ui/src/hooks/useAuth.ts`
- `ui/src/components/auth/LoginForm.tsx`
- `ui/src/components/auth/ApiKeyManager.tsx`
- `ui/src/pages/Login.tsx`

**Implementation Details:**

```typescript
// ui/src/contexts/AuthContext.tsx
interface AuthContextType {
  user: User | null;
  apiKey: string | null;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  setApiKey: (key: string) => void;
}

// Supports both JWT and API key authentication
// Stores tokens in localStorage with encryption
// Auto-refresh JWT tokens before expiry
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 1.1.1 | Create AuthContext | Implement auth state management | 3 |
| 1.1.2 | Create useAuth hook | Expose auth operations | 2 |
| 1.1.3 | Create LoginForm component | Username/password + API key login | 3 |
| 1.1.4 | Create ApiKeyManager | Generate, list, revoke API keys | 4 |
| 1.1.5 | Create Login page | Full login page with routing | 2 |
| 1.1.6 | Add protected routes | Route guards for authenticated pages | 2 |
| 1.1.7 | Write tests | Unit tests for auth components | 3 |

### 3.2 Theme System

**Files to create:**
- `ui/src/contexts/ThemeContext.tsx`
- `ui/src/hooks/useTheme.ts`
- `ui/src/styles/themes/light.css`
- `ui/src/styles/themes/dark.css`

**Implementation Details:**

```typescript
// CSS variables approach for easy theming
:root {
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f8fafc;
  --color-text-primary: #1e293b;
  --color-text-secondary: #64748b;
  --color-border: #e2e8f0;
  --color-accent: #3b82f6;
  --color-success: #22c55e;
  --color-warning: #eab308;
  --color-danger: #ef4444;
}

[data-theme="dark"] {
  --color-bg-primary: #0f172a;
  --color-bg-secondary: #1e293b;
  --color-text-primary: #f8fafc;
  --color-text-secondary: #94a3b8;
  --color-border: #334155;
}
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 1.2.1 | Create ThemeContext | Theme state and toggle | 2 |
| 1.2.2 | Define CSS variables | Light and dark theme variables | 3 |
| 1.2.3 | Update all components | Apply CSS variables | 4 |
| 1.2.4 | Add theme toggle UI | Button in header | 1 |
| 1.2.5 | Persist preference | Save to localStorage | 1 |

### 3.3 Layout Enhancements

**Files to modify:**
- `ui/src/components/Layout.tsx`

**Files to create:**
- `ui/src/components/layout/Breadcrumbs.tsx`
- `ui/src/components/layout/UserMenu.tsx`
- `ui/src/components/layout/Sidebar.tsx`
- `ui/src/components/layout/Header.tsx`
- `ui/src/components/NotificationCenter.tsx`

**Implementation Details:**

```typescript
// Layout structure
<Layout>
  <Header>
    <Logo />
    <Breadcrumbs />
    <SearchBar />
    <NotificationCenter />
    <ThemeToggle />
    <UserMenu />
  </Header>
  <Sidebar>
    <NavItem icon={Dashboard} to="/" />
    <NavItem icon={Traces} to="/traces" />
    <NavItem icon={Drift} to="/drift" />
    <NavItem icon={Costs} to="/costs" />
    <NavItem icon={Alerts} to="/alerts" />
    <NavItem icon={Intelligence} to="/intelligence" />
    <NavItem icon={RCA} to="/rca" />
    <NavItem icon={Settings} to="/settings" />
  </Sidebar>
  <Main>
    <Outlet />
  </Main>
</Layout>
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 1.3.1 | Create Breadcrumbs | Dynamic breadcrumb navigation | 2 |
| 1.3.2 | Create UserMenu | User dropdown with logout | 2 |
| 1.3.3 | Enhance Sidebar | Collapsible, icons, badges | 3 |
| 1.3.4 | Create Header | Responsive header component | 2 |
| 1.3.5 | Create NotificationCenter | Real-time notifications dropdown | 4 |
| 1.3.6 | Add keyboard shortcuts | Cmd+K search, navigation | 3 |
| 1.3.7 | Mobile responsive | Hamburger menu, touch support | 4 |

### 3.4 WebSocket Integration

**Files to create:**
- `ui/src/hooks/useWebSocket.ts`
- `ui/src/contexts/WebSocketContext.tsx`
- `ui/src/services/websocket.ts`

**Implementation Details:**

```typescript
// WebSocket service
class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private subscribers: Map<string, Set<(data: any) => void>> = new Map();

  connect(url: string): void;
  disconnect(): void;
  subscribe(channel: string, callback: (data: any) => void): () => void;
  send(message: WebSocketMessage): void;
}

// Channels:
// - traces:{model_id} - Real-time trace updates
// - alerts:* - All alerts
// - drift:{model_id} - Drift score updates
// - health:system - System health updates
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 1.4.1 | Create WebSocket service | Connection management, reconnect | 4 |
| 1.4.2 | Create WebSocketContext | Provider with connection state | 2 |
| 1.4.3 | Create useWebSocket hook | Subscribe to channels | 2 |
| 1.4.4 | Add connection indicator | UI for connection status | 1 |
| 1.4.5 | Write tests | Mock WebSocket tests | 3 |

---

## 4. Stage 2: Trace Explorer

### 4.1 Trace List Enhancements

**Files to modify:**
- `ui/src/pages/Traces.tsx`
- `ui/src/hooks/useTraces.ts`

**Files to create:**
- `ui/src/components/traces/TraceList.tsx`
- `ui/src/components/traces/TraceSearch.tsx`
- `ui/src/components/traces/TraceFilters.tsx`
- `ui/src/components/traces/SavedQueries.tsx`
- `ui/src/components/traces/BulkActions.tsx`

**Implementation Details:**

```typescript
// TraceSearch with query language
interface TraceQuery {
  service?: string;
  model_id?: string;
  status?: 'ok' | 'error';
  duration_min?: number;
  duration_max?: number;
  time_range: TimeRange;
  attributes?: Record<string, string>;
  has_drift?: boolean;
  has_safety_issues?: boolean;
  sort_by?: 'time' | 'duration' | 'cost';
  sort_order?: 'asc' | 'desc';
}

// Query language examples:
// service:my-service status:error duration:>1000ms
// model:gpt-4 has:drift time:last-1h
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 2.1.1 | Create TraceList | Virtualized list for performance | 4 |
| 2.1.2 | Create TraceSearch | Query language parser | 6 |
| 2.1.3 | Create TraceFilters | Filter sidebar with facets | 4 |
| 2.1.4 | Create SavedQueries | Save/load query presets | 3 |
| 2.1.5 | Create BulkActions | Select, export, compare | 3 |
| 2.1.6 | Add real-time updates | WebSocket integration | 3 |
| 2.1.7 | Add pagination/infinite scroll | Performance optimization | 2 |

### 4.2 Span Waterfall Visualization

**Files to create:**
- `ui/src/components/traces/SpanWaterfall.tsx`
- `ui/src/components/traces/SpanRow.tsx`
- `ui/src/components/traces/SpanDetail.tsx`
- `ui/src/components/traces/TimeRuler.tsx`

**Implementation Details:**

```typescript
// SpanWaterfall visualization
interface SpanWaterfallProps {
  spans: Span[];
  traceStart: number;
  traceDuration: number;
  selectedSpanId?: string;
  onSpanSelect: (spanId: string) => void;
  showCriticalPath?: boolean;
}

// Features:
// - Hierarchical span display with indentation
// - Time-relative positioning and width
// - Color coding by service/status
// - Critical path highlighting
// - Zoom and pan controls
// - Span search within trace
```

**Visual representation:**
```
Time Ruler: |----0ms----|----100ms----|----200ms----|----300ms----|
┌─────────────────────────────────────────────────────────────────┐
│ ▼ root-span (service-a)                                         │
│   [███████████████████████████████████████████████████] 300ms   │
│   ├─ ▼ child-span-1 (service-b)                                 │
│   │   [████████████████████] 150ms                              │
│   │   ├─ leaf-span (service-c)                                  │
│   │   │   [████████] 80ms                                       │
│   ├─ child-span-2 (service-a)                                   │
│   │   [██████] 50ms                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 2.2.1 | Create TimeRuler | Time scale with zoom | 3 |
| 2.2.2 | Create SpanRow | Individual span bar | 4 |
| 2.2.3 | Create SpanWaterfall | Container with hierarchy | 6 |
| 2.2.4 | Create SpanDetail | Selected span details panel | 4 |
| 2.2.5 | Add critical path | Highlight slowest path | 3 |
| 2.2.6 | Add zoom/pan controls | Interactive navigation | 3 |
| 2.2.7 | Add span search | Find spans within trace | 2 |
| 2.2.8 | Write tests | Visual regression tests | 3 |

### 4.3 Trace Comparison

**Files to create:**
- `ui/src/components/traces/TraceComparison.tsx`
- `ui/src/components/traces/DiffViewer.tsx`
- `ui/src/pages/TraceCompare.tsx`

**Implementation Details:**

```typescript
// Side-by-side trace comparison
interface TraceComparisonProps {
  leftTrace: Trace;
  rightTrace: Trace;
  diffMode: 'structure' | 'timing' | 'attributes';
}

// Features:
// - Side-by-side waterfall views
// - Structural diff (added/removed spans)
// - Timing diff (highlight slower/faster)
// - Attribute diff (changed values)
// - Summary statistics comparison
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 2.3.1 | Create TraceComparison | Split view layout | 4 |
| 2.3.2 | Create DiffViewer | Diff algorithm and display | 6 |
| 2.3.3 | Create TraceCompare page | Full comparison page | 3 |
| 2.3.4 | Add comparison from list | Select and compare UI | 2 |
| 2.3.5 | Add comparison summary | Statistics comparison | 2 |

### 4.4 Real-time Trace Streaming

**Files to create:**
- `ui/src/components/traces/LiveTraceStream.tsx`
- `ui/src/hooks/useLiveTraces.ts`

**Implementation Details:**

```typescript
// Live trace streaming with filtering
interface LiveTraceStreamProps {
  filters: TraceQuery;
  maxTraces?: number;  // Rolling buffer
  isPaused?: boolean;
  onTraceClick: (traceId: string) => void;
}

// Features:
// - WebSocket subscription for real-time traces
// - Rolling buffer (last N traces)
// - Pause/resume functionality
// - Filter updates without reconnect
// - Highlight errors and anomalies
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 2.4.1 | Create useLiveTraces hook | WebSocket subscription | 3 |
| 2.4.2 | Create LiveTraceStream | Streaming list UI | 4 |
| 2.4.3 | Add pause/resume | Control live feed | 1 |
| 2.4.4 | Add filtering | Real-time filter application | 2 |
| 2.4.5 | Add anomaly highlight | Visual indicators | 2 |

---

## 5. Stage 3: Drift Dashboard

### 5.1 Drift Overview Enhancements

**Files to modify:**
- `ui/src/pages/Drift.tsx`
- `ui/src/components/DriftHeatmap.tsx`

**Files to create:**
- `ui/src/components/drift/DriftOverview.tsx`
- `ui/src/components/drift/DriftTimeline.tsx`
- `ui/src/components/drift/DriftAlertBanner.tsx`
- `ui/src/hooks/useDriftHistory.ts`

**Implementation Details:**

```typescript
// Drift overview with multi-dimensional view
interface DriftOverviewProps {
  modelId: string;
  timeRange: TimeRange;
  driftTypes: DriftType[];
}

// Dashboard sections:
// 1. Overall drift status (healthy/warning/critical)
// 2. Drift score by type (feature, embedding, concept, prediction)
// 3. Timeline of drift events
// 4. Active drift alerts
// 5. Affected features heatmap
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 3.1.1 | Create DriftOverview | Multi-section layout | 4 |
| 3.1.2 | Create DriftTimeline | Time series visualization | 5 |
| 3.1.3 | Create DriftAlertBanner | Active alert display | 2 |
| 3.1.4 | Create useDriftHistory | Historical data fetching | 3 |
| 3.1.5 | Enhance DriftHeatmap | Interactive drill-down | 4 |

### 5.2 Feature-Level Drift Breakdown

**Files to create:**
- `ui/src/components/drift/FeatureDriftBreakdown.tsx`
- `ui/src/components/drift/FeatureDriftCard.tsx`
- `ui/src/components/drift/DistributionChart.tsx`

**Implementation Details:**

```typescript
// Feature-level drift analysis
interface FeatureDriftBreakdownProps {
  modelId: string;
  features: FeatureDrift[];
  onFeatureSelect: (featureName: string) => void;
}

interface FeatureDrift {
  name: string;
  type: 'numerical' | 'categorical' | 'embedding';
  driftScore: number;
  pValue: number;
  referenceDistribution: Distribution;
  currentDistribution: Distribution;
  trend: 'stable' | 'increasing' | 'decreasing';
}

// Visualization:
// - Sortable list by drift score
// - Distribution comparison charts
// - Statistical test results
// - Trend indicators
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 3.2.1 | Create FeatureDriftBreakdown | Feature list with sorting | 4 |
| 3.2.2 | Create FeatureDriftCard | Individual feature card | 3 |
| 3.2.3 | Create DistributionChart | Reference vs current | 5 |
| 3.2.4 | Add statistical details | P-values, test results | 2 |
| 3.2.5 | Add export functionality | CSV/JSON export | 2 |

### 5.3 Drift Alerts Integration

**Files to create:**
- `ui/src/components/drift/DriftAlertConfig.tsx`
- `ui/src/components/drift/DriftAlertHistory.tsx`

**Implementation Details:**

```typescript
// Configure drift alert thresholds
interface DriftAlertConfigProps {
  modelId: string;
  currentConfig: DriftAlertConfig;
  onSave: (config: DriftAlertConfig) => Promise<void>;
}

interface DriftAlertConfig {
  enabled: boolean;
  thresholds: {
    feature: number;      // e.g., 0.3
    embedding: number;    // e.g., 0.2
    concept: number;      // e.g., 0.4
    prediction: number;   // e.g., 0.3
  };
  evaluationWindow: string;  // e.g., "1h"
  notificationChannels: string[];
}
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 3.3.1 | Create DriftAlertConfig | Configuration form | 4 |
| 3.3.2 | Create DriftAlertHistory | Historical alerts list | 3 |
| 3.3.3 | Add threshold visualization | Threshold lines on charts | 2 |
| 3.3.4 | Add alert test | Test alert configuration | 2 |

---

## 6. Stage 4: Cost Analytics

### 6.1 Cost Overview Enhancements

**Files to modify:**
- `ui/src/pages/Costs.tsx`
- `ui/src/components/CostCharts.tsx`

**Files to create:**
- `ui/src/components/costs/CostOverview.tsx`
- `ui/src/components/costs/CostBreakdownTable.tsx`
- `ui/src/components/costs/TokenUsageChart.tsx`
- `ui/src/hooks/useCostAnalytics.ts`

**Implementation Details:**

```typescript
// Cost overview dashboard
interface CostOverviewProps {
  timeRange: TimeRange;
  groupBy: 'model' | 'user' | 'feature' | 'team';
}

// Dashboard sections:
// 1. Total cost with trend
// 2. Cost breakdown by dimension
// 3. Token usage (input vs output)
// 4. Cost per request trend
// 5. Top cost drivers
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 4.1.1 | Create CostOverview | Multi-section layout | 4 |
| 4.1.2 | Create CostBreakdownTable | Sortable breakdown table | 3 |
| 4.1.3 | Create TokenUsageChart | Stacked bar chart | 3 |
| 4.1.4 | Create useCostAnalytics | Analytics data fetching | 3 |
| 4.1.5 | Add dimension switcher | Group by selector | 2 |

### 6.2 Budget Tracking

**Files to create:**
- `ui/src/components/costs/BudgetTracker.tsx`
- `ui/src/components/costs/BudgetConfig.tsx`
- `ui/src/components/costs/BudgetAlert.tsx`

**Implementation Details:**

```typescript
// Budget tracking with alerts
interface BudgetTrackerProps {
  budgets: Budget[];
  currentPeriod: string;
}

interface Budget {
  id: string;
  name: string;
  dimension: 'model' | 'user' | 'feature' | 'team' | 'global';
  dimensionValue?: string;
  limit: number;
  period: 'daily' | 'weekly' | 'monthly';
  current: number;
  alertThreshold: number;  // e.g., 0.8 for 80%
}

// Visualization:
// - Progress bars with thresholds
// - Alert indicators
// - Projected overage
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 4.2.1 | Create BudgetTracker | Budget progress display | 4 |
| 4.2.2 | Create BudgetConfig | Budget CRUD interface | 5 |
| 4.2.3 | Create BudgetAlert | Alert banner and notifications | 3 |
| 4.2.4 | Add budget projections | Forecasted usage vs budget | 3 |

### 6.3 Cost Forecasting

**Files to create:**
- `ui/src/components/costs/CostForecast.tsx`
- `ui/src/components/costs/ForecastChart.tsx`
- `ui/src/hooks/useCostForecast.ts`

**Implementation Details:**

```typescript
// Cost forecasting with confidence intervals
interface CostForecastProps {
  modelId?: string;
  horizonDays: number;
  showConfidenceInterval?: boolean;
}

// Features:
// - Historical trend line
// - Forecast line with confidence band
// - Scenario comparison (current vs optimized)
// - Anomaly detection in forecast
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 4.3.1 | Create CostForecast | Forecast display container | 3 |
| 4.3.2 | Create ForecastChart | Line chart with bands | 5 |
| 4.3.3 | Create useCostForecast | Forecast API integration | 3 |
| 4.3.4 | Add scenario comparison | What-if analysis | 4 |

### 6.4 Cost Export and Reports

**Files to create:**
- `ui/src/components/costs/CostExport.tsx`
- `ui/src/components/costs/CostReport.tsx`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 4.4.1 | Create CostExport | CSV/PDF export dialog | 3 |
| 4.4.2 | Create CostReport | Printable report view | 4 |
| 4.4.3 | Add scheduled reports | Email report configuration | 3 |

---

## 7. Stage 5: SDK Integration Enhancement

### 7.1 OpenAI Integration Enhancement

**Files to modify:**
- `sdk/python/pyflare/integrations/openai.py`

**Files to create:**
- `sdk/python/pyflare/integrations/openai_streaming.py`
- `sdk/python/tests/test_openai_integration.py`

**Implementation Details:**

```python
# Enhanced OpenAI integration
class OpenAIInstrumentation:
    """Auto-instrument OpenAI API calls with PyFlare tracing."""

    def instrument(self):
        """Patch OpenAI client methods."""
        # Patch: chat.completions.create
        # Patch: embeddings.create
        # Patch: completions.create (legacy)

    def _trace_chat_completion(self, original_func):
        """Trace chat completions with streaming support."""
        @functools.wraps(original_func)
        async def wrapper(*args, **kwargs):
            with self.tracer.start_span("openai.chat") as span:
                span.set_attribute("model", kwargs.get("model"))
                span.set_attribute("messages.count", len(kwargs.get("messages", [])))

                if kwargs.get("stream"):
                    return self._trace_stream(span, original_func, *args, **kwargs)
                else:
                    response = await original_func(*args, **kwargs)
                    span.set_attribute("tokens.input", response.usage.prompt_tokens)
                    span.set_attribute("tokens.output", response.usage.completion_tokens)
                    return response
        return wrapper

    def _trace_stream(self, span, func, *args, **kwargs):
        """Handle streaming responses with proper token counting."""
        # Yield chunks while accumulating for final span attributes

    def _trace_function_calls(self, span, tool_calls):
        """Create child spans for function/tool calls."""
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 5.1.1 | Add streaming support | Stream response tracing | 6 |
| 5.1.2 | Add function call tracing | Tool use child spans | 4 |
| 5.1.3 | Add error categorization | API error classification | 3 |
| 5.1.4 | Add cost estimation | Real-time cost tracking | 3 |
| 5.1.5 | Write comprehensive tests | Unit and integration tests | 4 |
| 5.1.6 | Write documentation | Usage examples and API docs | 3 |

### 7.2 LangChain Integration Enhancement

**Files to modify:**
- `sdk/python/pyflare/integrations/langchain.py`

**Files to create:**
- `sdk/python/pyflare/integrations/langchain_agents.py`
- `sdk/python/pyflare/integrations/langchain_chains.py`
- `sdk/python/tests/test_langchain_integration.py`

**Implementation Details:**

```python
# Enhanced LangChain integration
class LangChainInstrumentation:
    """Comprehensive LangChain tracing for chains, agents, and tools."""

    def instrument(self):
        """Register callbacks for all LangChain components."""

    def trace_chain(self, chain):
        """Add tracing to a chain instance."""

    def trace_agent(self, agent):
        """Add tracing to an agent with tool calls."""

class PyFlareCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for PyFlare tracing."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Start span for chain execution."""

    def on_chain_end(self, outputs, **kwargs):
        """End span with outputs."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Start child span for tool execution."""

    def on_agent_action(self, action, **kwargs):
        """Record agent action decision."""

    def on_retriever_start(self, serialized, query, **kwargs):
        """Start span for retrieval."""
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 5.2.1 | Create callback handler | Full callback implementation | 6 |
| 5.2.2 | Add agent tracing | Multi-step agent spans | 5 |
| 5.2.3 | Add chain visualization | Chain structure in traces | 4 |
| 5.2.4 | Add retriever tracing | RAG pipeline visibility | 4 |
| 5.2.5 | Add memory tracing | Conversation memory tracking | 3 |
| 5.2.6 | Write tests | Comprehensive test suite | 4 |
| 5.2.7 | Write documentation | Usage guide and examples | 3 |

### 7.3 PyTorch Integration Enhancement

**Files to modify:**
- `sdk/python/pyflare/integrations/pytorch.py`

**Files to create:**
- `sdk/python/pyflare/integrations/pytorch_profiling.py`
- `sdk/python/tests/test_pytorch_integration.py`

**Implementation Details:**

```python
# Enhanced PyTorch integration
class PyTorchInstrumentation:
    """PyTorch model tracing with profiling support."""

    def instrument_model(self, model: nn.Module, name: str = None):
        """Wrap model forward pass with tracing."""

    def trace_inference(self, model, inputs):
        """Trace a single inference with detailed metrics."""
        with self.tracer.start_span("pytorch.inference") as span:
            span.set_attribute("model.name", model.__class__.__name__)
            span.set_attribute("model.parameters", count_parameters(model))

            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                output = model(inputs)

            span.set_attribute("compute.cpu_time_ms", prof.total_average().cpu_time_total / 1000)
            span.set_attribute("compute.cuda_time_ms", prof.total_average().cuda_time_total / 1000)
            span.set_attribute("memory.peak_mb", get_peak_memory_mb())

            return output

    def trace_training_step(self, model, batch, loss_fn, optimizer):
        """Trace a training step with gradients."""
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 5.3.1 | Add profiling integration | CPU/GPU profiling | 6 |
| 5.3.2 | Add gradient tracking | Gradient statistics | 4 |
| 5.3.3 | Add memory tracking | Memory usage metrics | 3 |
| 5.3.4 | Add batch statistics | Input/output shapes | 2 |
| 5.3.5 | Write tests | Model tracing tests | 4 |
| 5.3.6 | Write documentation | PyTorch usage guide | 3 |

### 7.4 PyFlame Native Integration

**Files to modify:**
- `sdk/python/pyflare/integrations/pyflame.py`

**Implementation Details:**

```python
# Native PyFlame integration
class PyFlameInstrumentation:
    """Native integration with PyFlame training framework."""

    def instrument_trainer(self, trainer):
        """Add tracing to PyFlame trainer."""

    def trace_cerebras_call(self, func):
        """Trace Cerebras accelerator calls."""

    def sync_metrics(self):
        """Sync PyFlame metrics to PyFlare."""
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 5.4.1 | Add trainer integration | Trainer callback hooks | 4 |
| 5.4.2 | Add Cerebras tracing | Hardware-specific spans | 4 |
| 5.4.3 | Add metric sync | PyFlame → PyFlare metrics | 3 |
| 5.4.4 | Write tests | Integration tests | 3 |
| 5.4.5 | Write documentation | PyFlame guide | 2 |

### 7.5 Integration Examples

**Files to create:**
- `examples/openai_chat.py`
- `examples/openai_streaming.py`
- `examples/langchain_agent.py`
- `examples/langchain_rag.py`
- `examples/pytorch_inference.py`
- `examples/pytorch_training.py`
- `examples/multi_model_pipeline.py`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 5.5.1 | Create OpenAI examples | Chat and streaming examples | 3 |
| 5.5.2 | Create LangChain examples | Agent and RAG examples | 4 |
| 5.5.3 | Create PyTorch examples | Inference and training | 3 |
| 5.5.4 | Create multi-model example | Complex pipeline example | 3 |

---

## 8. Stage 6: Grafana Plugin

### 8.1 Data Source Plugin

**Files to create:**
- `grafana-plugin/src/datasource.ts`
- `grafana-plugin/src/ConfigEditor.tsx`
- `grafana-plugin/src/QueryEditor.tsx`
- `grafana-plugin/src/types.ts`
- `grafana-plugin/plugin.json`

**Implementation Details:**

```typescript
// PyFlare Grafana data source
export class PyFlareDataSource extends DataSourceApi<PyFlareQuery, PyFlareDataSourceOptions> {
  constructor(instanceSettings: DataSourceInstanceSettings<PyFlareDataSourceOptions>) {
    super(instanceSettings);
  }

  async query(options: DataQueryRequest<PyFlareQuery>): Promise<DataQueryResponse> {
    // Support query types:
    // - traces: Trace data with span details
    // - metrics: Time series metrics
    // - drift: Drift scores over time
    // - costs: Cost data by dimension
    // - alerts: Alert timeline
  }

  async testDatasource(): Promise<any> {
    // Test connection to PyFlare API
  }
}

// Query types
interface PyFlareQuery extends DataQuery {
  queryType: 'traces' | 'metrics' | 'drift' | 'costs' | 'alerts';
  modelId?: string;
  metricName?: string;
  aggregation?: 'sum' | 'avg' | 'count' | 'p50' | 'p99';
}
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 6.1.1 | Initialize plugin project | Grafana plugin scaffolding | 2 |
| 6.1.2 | Create DataSource class | API integration | 6 |
| 6.1.3 | Create ConfigEditor | Connection settings UI | 3 |
| 6.1.4 | Create QueryEditor | Query builder UI | 5 |
| 6.1.5 | Add trace query support | Trace data fetching | 4 |
| 6.1.6 | Add metrics query support | Metrics time series | 4 |
| 6.1.7 | Add drift query support | Drift scores | 3 |
| 6.1.8 | Add cost query support | Cost data | 3 |
| 6.1.9 | Write tests | Plugin tests | 4 |

### 8.2 Trace Panel Plugin

**Files to create:**
- `grafana-plugin/src/panels/trace/TracePanel.tsx`
- `grafana-plugin/src/panels/trace/module.ts`

**Implementation Details:**

```typescript
// Trace visualization panel for Grafana
interface TracePanelOptions {
  showServiceColumn: boolean;
  showDurationColumn: boolean;
  showStatusColumn: boolean;
  maxTraces: number;
}

// Features:
// - Trace list view
// - Mini waterfall visualization
// - Click to open in PyFlare UI
// - Status/error highlighting
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 6.2.1 | Create TracePanel | Trace list panel | 5 |
| 6.2.2 | Add mini waterfall | Compact span view | 4 |
| 6.2.3 | Add linking | Deep link to PyFlare | 2 |
| 6.2.4 | Add panel options | Configuration UI | 2 |

### 8.3 Drift Panel Plugin

**Files to create:**
- `grafana-plugin/src/panels/drift/DriftPanel.tsx`
- `grafana-plugin/src/panels/drift/module.ts`

**Implementation Details:**

```typescript
// Drift visualization panel for Grafana
interface DriftPanelOptions {
  showThreshold: boolean;
  driftTypes: DriftType[];
  alertOnDrift: boolean;
}

// Features:
// - Drift score gauge
// - Trend sparkline
// - Multi-type comparison
// - Alert indicator
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 6.3.1 | Create DriftPanel | Drift gauge panel | 4 |
| 6.3.2 | Add heatmap view | Feature drift heatmap | 5 |
| 6.3.3 | Add alert integration | Grafana alerting | 3 |

### 8.4 Plugin Documentation and Distribution

**Files to create:**
- `grafana-plugin/README.md`
- `grafana-plugin/CHANGELOG.md`
- `grafana-plugin/docs/installation.md`
- `grafana-plugin/docs/configuration.md`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 6.4.1 | Write installation guide | Setup instructions | 2 |
| 6.4.2 | Write configuration guide | Data source setup | 2 |
| 6.4.3 | Create example dashboards | JSON dashboard templates | 4 |
| 6.4.4 | Sign plugin | Grafana plugin signing | 2 |
| 6.4.5 | Publish to marketplace | Grafana plugin catalog | 2 |

---

## 9. Stage 7: Documentation

### 9.1 API Reference Documentation

**Files to create:**
- `docs/api-reference/overview.md`
- `docs/api-reference/authentication.md`
- `docs/api-reference/traces.md`
- `docs/api-reference/drift.md`
- `docs/api-reference/costs.md`
- `docs/api-reference/alerts.md`
- `docs/api-reference/intelligence.md`
- `docs/api-reference/rca.md`
- `docs/api-reference/query.md`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 7.1.1 | Document authentication | Auth methods and examples | 3 |
| 7.1.2 | Document traces API | All trace endpoints | 4 |
| 7.1.3 | Document drift API | Drift endpoints | 3 |
| 7.1.4 | Document costs API | Cost endpoints | 3 |
| 7.1.5 | Document alerts API | Alert endpoints | 4 |
| 7.1.6 | Document intelligence API | Intelligence endpoints | 3 |
| 7.1.7 | Document RCA API | RCA endpoints | 3 |
| 7.1.8 | Generate OpenAPI spec | Auto-generate from code | 4 |

### 9.2 Getting Started Guide

**Files to create:**
- `docs/getting-started/quickstart.md`
- `docs/getting-started/installation.md`
- `docs/getting-started/first-trace.md`
- `docs/getting-started/python-sdk.md`
- `docs/getting-started/ui-tour.md`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 7.2.1 | Write quickstart | 5-minute getting started | 3 |
| 7.2.2 | Write installation guide | All installation methods | 4 |
| 7.2.3 | Write first trace tutorial | Hello world trace | 3 |
| 7.2.4 | Write SDK guide | Python SDK tutorial | 4 |
| 7.2.5 | Write UI tour | Feature walkthrough | 3 |

### 9.3 Deployment Guide

**Files to create:**
- `docs/deployment/docker-compose.md`
- `docs/deployment/kubernetes.md`
- `docs/deployment/production-checklist.md`
- `docs/deployment/scaling.md`
- `docs/deployment/security.md`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 7.3.1 | Write Docker Compose guide | Local deployment | 3 |
| 7.3.2 | Write Kubernetes guide | Production deployment | 6 |
| 7.3.3 | Write production checklist | Pre-production checklist | 3 |
| 7.3.4 | Write scaling guide | Horizontal scaling | 4 |
| 7.3.5 | Write security guide | Security best practices | 4 |

### 9.4 Architecture Documentation

**Files to create:**
- `docs/architecture/overview.md`
- `docs/architecture/data-flow.md`
- `docs/architecture/components.md`
- `docs/architecture/storage.md`
- `docs/architecture/processing.md`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 7.4.1 | Write architecture overview | System design | 4 |
| 7.4.2 | Document data flow | End-to-end flow | 3 |
| 7.4.3 | Document components | Component deep dives | 5 |
| 7.4.4 | Document storage | Schema and design | 3 |
| 7.4.5 | Document processing | Pipeline details | 4 |

### 9.5 SDK Documentation

**Files to create:**
- `docs/sdk/python/overview.md`
- `docs/sdk/python/configuration.md`
- `docs/sdk/python/decorators.md`
- `docs/sdk/python/integrations.md`
- `docs/sdk/python/advanced.md`

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 7.5.1 | Write SDK overview | SDK introduction | 3 |
| 7.5.2 | Write configuration guide | SDK configuration | 3 |
| 7.5.3 | Write decorator guide | Decorator usage | 3 |
| 7.5.4 | Write integrations guide | All integrations | 5 |
| 7.5.5 | Write advanced guide | Advanced patterns | 4 |

---

## 10. Stage 8: Testing & Performance

### 10.1 End-to-End Test Suite

**Files to create:**
- `tests/e2e/setup.ts`
- `tests/e2e/traces.spec.ts`
- `tests/e2e/drift.spec.ts`
- `tests/e2e/costs.spec.ts`
- `tests/e2e/alerts.spec.ts`
- `tests/e2e/navigation.spec.ts`

**Implementation Details:**

```typescript
// Playwright E2E tests
import { test, expect } from '@playwright/test';

test.describe('Trace Explorer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/traces');
  });

  test('should display trace list', async ({ page }) => {
    await expect(page.locator('[data-testid="trace-list"]')).toBeVisible();
  });

  test('should filter traces by service', async ({ page }) => {
    await page.fill('[data-testid="service-filter"]', 'my-service');
    await expect(page.locator('[data-testid="trace-row"]')).toHaveCount(10);
  });

  test('should open trace detail', async ({ page }) => {
    await page.click('[data-testid="trace-row"]:first-child');
    await expect(page.url()).toContain('/traces/');
    await expect(page.locator('[data-testid="span-waterfall"]')).toBeVisible();
  });
});
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 8.1.1 | Set up Playwright | E2E test infrastructure | 3 |
| 8.1.2 | Write trace tests | Trace explorer E2E tests | 4 |
| 8.1.3 | Write drift tests | Drift dashboard E2E tests | 3 |
| 8.1.4 | Write cost tests | Cost analytics E2E tests | 3 |
| 8.1.5 | Write alert tests | Alert management E2E tests | 3 |
| 8.1.6 | Write navigation tests | Navigation and auth tests | 2 |
| 8.1.7 | Add CI integration | GitHub Actions E2E | 2 |

### 10.2 Performance Benchmarks

**Files to create:**
- `benchmarks/ingestion_benchmark.cpp`
- `benchmarks/query_benchmark.cpp`
- `benchmarks/drift_benchmark.cpp`
- `benchmarks/run_benchmarks.sh`
- `benchmarks/results/README.md`

**Implementation Details:**

```cpp
// Benchmark categories
// 1. Ingestion throughput
// - OTLP receiver: traces/second
// - Kafka producer: messages/second
// - Storage writer: rows/second

// 2. Query latency
// - Trace retrieval: p50, p99
// - Aggregation queries: p50, p99
// - Full-text search: p50, p99

// 3. Processing throughput
// - Drift detection: evaluations/second
// - Cost calculation: traces/second
// - Alert evaluation: rules/second
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 8.2.1 | Write ingestion benchmarks | OTLP, Kafka, storage | 6 |
| 8.2.2 | Write query benchmarks | Query performance | 5 |
| 8.2.3 | Write processing benchmarks | Drift, cost, alert | 5 |
| 8.2.4 | Create benchmark runner | Automated benchmark script | 3 |
| 8.2.5 | Document baseline | Record baseline metrics | 2 |
| 8.2.6 | Add CI benchmarks | Regression detection | 3 |

### 10.3 Load Testing

**Files to create:**
- `tests/load/k6/traces.js`
- `tests/load/k6/queries.js`
- `tests/load/k6/mixed_workload.js`
- `tests/load/docker-compose.load.yml`

**Implementation Details:**

```javascript
// k6 load test for trace ingestion
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 100 },   // Ramp up
    { duration: '5m', target: 100 },   // Steady state
    { duration: '1m', target: 500 },   // Spike
    { duration: '5m', target: 500 },   // High load
    { duration: '1m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<500'],  // 99% under 500ms
    http_req_failed: ['rate<0.01'],    // <1% errors
  },
};

export default function () {
  const payload = generateTracePayload();
  const res = http.post('http://localhost:4318/v1/traces', payload);
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
  sleep(0.1);
}
```

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 8.3.1 | Write trace load tests | Ingestion load test | 4 |
| 8.3.2 | Write query load tests | Query load test | 4 |
| 8.3.3 | Write mixed workload test | Realistic workload | 4 |
| 8.3.4 | Create load test environment | Docker Compose setup | 3 |
| 8.3.5 | Document load test results | Performance report | 3 |

### 10.4 Security Audit

**Tasks:**
| ID | Task | Description | Est. Hours |
|----|------|-------------|------------|
| 8.4.1 | Run SAST scan | Static analysis | 2 |
| 8.4.2 | Run dependency scan | Vulnerability check | 2 |
| 8.4.3 | Run OWASP ZAP scan | Web security scan | 3 |
| 8.4.4 | Review auth implementation | Auth security review | 4 |
| 8.4.5 | Review API security | Input validation, etc. | 4 |
| 8.4.6 | Document findings | Security report | 3 |

---

## 11. Implementation Order & Dependencies

### Critical Path

```
Stage 1 (Foundation) ─┬─► Stage 2 (Traces) ─────┬─► Stage 8 (Testing)
                      │                         │
                      ├─► Stage 3 (Drift) ──────┤
                      │                         │
                      ├─► Stage 4 (Costs) ──────┤
                      │                         │
                      └─► Stage 5 (SDK) ────────┘
                           │
                           └─► Stage 6 (Grafana)
                               │
                               └─► Stage 7 (Docs)
```

### Recommended Implementation Order

**Week 1: Foundation + Trace Explorer Start**
- Stage 1: All tasks (Authentication, Theme, Layout, WebSocket)
- Stage 2: Tasks 2.1.1-2.1.4 (Trace list enhancements)

**Week 2: Trace Explorer + Drift Dashboard**
- Stage 2: Tasks 2.2.1-2.4.5 (Waterfall, Comparison, Streaming)
- Stage 3: Tasks 3.1.1-3.2.5 (Drift overview, Feature breakdown)

**Week 3: Cost Analytics + SDK + Grafana Start**
- Stage 4: All tasks (Cost overview, Budget, Forecast)
- Stage 5: Tasks 5.1.1-5.2.7 (OpenAI, LangChain enhancements)
- Stage 6: Tasks 6.1.1-6.1.5 (Grafana data source)

**Week 4: Grafana + Documentation + Testing**
- Stage 5: Tasks 5.3.1-5.5.4 (PyTorch, PyFlame, Examples)
- Stage 6: Tasks 6.1.6-6.4.5 (Grafana panels, Distribution)
- Stage 7: All tasks (Documentation)
- Stage 8: All tasks (Testing, Performance, Security)

---

## 12. Verification Checklist

### Pre-Release Verification

**UI Functionality:**
- [ ] Authentication flow works (login, logout, API key)
- [ ] All pages load without errors
- [ ] Dark/light theme toggle works
- [ ] Real-time updates via WebSocket
- [ ] Mobile responsive design
- [ ] Keyboard shortcuts work

**Trace Explorer:**
- [ ] Trace list displays with pagination
- [ ] Search and filters work correctly
- [ ] Span waterfall renders properly
- [ ] Trace comparison works
- [ ] Real-time streaming works

**Drift Dashboard:**
- [ ] Drift scores display correctly
- [ ] Timeline visualization works
- [ ] Feature breakdown is accurate
- [ ] Alert configuration saves

**Cost Analytics:**
- [ ] Cost overview displays correctly
- [ ] Budget tracking works
- [ ] Forecasting displays
- [ ] Export generates valid files

**SDK Integrations:**
- [ ] OpenAI integration tests pass
- [ ] LangChain integration tests pass
- [ ] PyTorch integration tests pass
- [ ] Examples run successfully

**Grafana Plugin:**
- [ ] Data source connects
- [ ] All query types work
- [ ] Panels render correctly
- [ ] Plugin passes signing

**Documentation:**
- [ ] API reference is complete
- [ ] Getting started guide works
- [ ] Deployment guide is accurate
- [ ] SDK documentation is current

**Testing:**
- [ ] E2E tests pass
- [ ] Performance benchmarks meet targets
- [ ] Load tests pass thresholds
- [ ] Security audit has no critical findings

### Performance Targets

| Metric | Target |
|--------|--------|
| UI initial load | < 2s |
| Page navigation | < 500ms |
| Trace list render | < 1s (1000 traces) |
| Waterfall render | < 500ms (100 spans) |
| API response (p99) | < 500ms |
| WebSocket latency | < 100ms |

---

## Appendix A: File Index

### New UI Files (Stage 1-4)

```
ui/src/
├── contexts/
│   ├── AuthContext.tsx
│   ├── ThemeContext.tsx
│   └── WebSocketContext.tsx
├── hooks/
│   ├── useAuth.ts
│   ├── useTheme.ts
│   ├── useWebSocket.ts
│   ├── useLiveTraces.ts
│   ├── useDriftHistory.ts
│   ├── useCostAnalytics.ts
│   └── useCostForecast.ts
├── components/
│   ├── auth/
│   │   ├── LoginForm.tsx
│   │   └── ApiKeyManager.tsx
│   ├── layout/
│   │   ├── Breadcrumbs.tsx
│   │   ├── UserMenu.tsx
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   ├── traces/
│   │   ├── TraceList.tsx
│   │   ├── TraceSearch.tsx
│   │   ├── TraceFilters.tsx
│   │   ├── SavedQueries.tsx
│   │   ├── BulkActions.tsx
│   │   ├── SpanWaterfall.tsx
│   │   ├── SpanRow.tsx
│   │   ├── SpanDetail.tsx
│   │   ├── TimeRuler.tsx
│   │   ├── TraceComparison.tsx
│   │   ├── DiffViewer.tsx
│   │   └── LiveTraceStream.tsx
│   ├── drift/
│   │   ├── DriftOverview.tsx
│   │   ├── DriftTimeline.tsx
│   │   ├── DriftAlertBanner.tsx
│   │   ├── FeatureDriftBreakdown.tsx
│   │   ├── FeatureDriftCard.tsx
│   │   ├── DistributionChart.tsx
│   │   ├── DriftAlertConfig.tsx
│   │   └── DriftAlertHistory.tsx
│   └── costs/
│       ├── CostOverview.tsx
│       ├── CostBreakdownTable.tsx
│       ├── TokenUsageChart.tsx
│       ├── BudgetTracker.tsx
│       ├── BudgetConfig.tsx
│       ├── BudgetAlert.tsx
│       ├── CostForecast.tsx
│       ├── ForecastChart.tsx
│       ├── CostExport.tsx
│       └── CostReport.tsx
├── pages/
│   ├── Login.tsx
│   └── TraceCompare.tsx
├── services/
│   └── websocket.ts
└── styles/
    └── themes/
        ├── light.css
        └── dark.css
```

### Grafana Plugin Files (Stage 6)

```
grafana-plugin/
├── src/
│   ├── datasource.ts
│   ├── ConfigEditor.tsx
│   ├── QueryEditor.tsx
│   ├── types.ts
│   └── panels/
│       ├── trace/
│       │   ├── TracePanel.tsx
│       │   └── module.ts
│       └── drift/
│           ├── DriftPanel.tsx
│           └── module.ts
├── plugin.json
├── package.json
├── README.md
└── docs/
    ├── installation.md
    └── configuration.md
```

### Documentation Files (Stage 7)

```
docs/
├── api-reference/
│   ├── overview.md
│   ├── authentication.md
│   ├── traces.md
│   ├── drift.md
│   ├── costs.md
│   ├── alerts.md
│   ├── intelligence.md
│   ├── rca.md
│   └── query.md
├── getting-started/
│   ├── quickstart.md
│   ├── installation.md
│   ├── first-trace.md
│   ├── python-sdk.md
│   └── ui-tour.md
├── deployment/
│   ├── docker-compose.md
│   ├── kubernetes.md
│   ├── production-checklist.md
│   ├── scaling.md
│   └── security.md
├── architecture/
│   ├── overview.md
│   ├── data-flow.md
│   ├── components.md
│   ├── storage.md
│   └── processing.md
└── sdk/
    └── python/
        ├── overview.md
        ├── configuration.md
        ├── decorators.md
        ├── integrations.md
        └── advanced.md
```

---

*This plan is the authoritative guide for Phase 4 development. Update as implementation progresses.*
