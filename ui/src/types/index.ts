// PyFlare UI Type Definitions

export interface Trace {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  serviceName: string;
  operationName: string;
  modelId: string;
  modelVersion: string;
  status: 'ok' | 'error';
  startTime: string;
  endTime: string;
  durationMs: number;
  inputPreview?: string;
  outputPreview?: string;
  inputTokens?: number;
  outputTokens?: number;
  costMicros?: number;
  attributes: Record<string, string>;
}

export interface Span {
  spanId: string;
  parentSpanId?: string;
  operationName: string;
  startTime: string;
  endTime: string;
  durationMs: number;
  status: 'ok' | 'error';
  attributes: Record<string, string>;
  events: SpanEvent[];
}

export interface SpanEvent {
  name: string;
  timestamp: string;
  attributes: Record<string, string>;
}

export interface DriftStatus {
  modelId: string;
  overallStatus: 'healthy' | 'warning' | 'drifted';
  driftScores: {
    feature?: DriftScore;
    embedding?: DriftScore;
    concept?: DriftScore;
    prediction?: DriftScore;
  };
  lastUpdated: string;
}

export interface DriftScore {
  score: number;
  threshold: number;
  isDrifted: boolean;
  trend: 'stable' | 'increasing' | 'decreasing';
  featureBreakdown?: Record<string, number>;
}

export interface CostSummary {
  totalCostUsd: number;
  totalRequests: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  averageCostPerRequestUsd: number;
  byPeriod: {
    period: string;
    costUsd: number;
    requests: number;
  }[];
}

export interface Alert {
  alertId: string;
  alertType: 'drift' | 'cost' | 'error_rate' | 'latency' | 'custom';
  severity: 'critical' | 'warning' | 'info';
  status: 'firing' | 'resolved' | 'acknowledged';
  modelId?: string;
  serviceName?: string;
  title: string;
  description: string;
  triggeredAt: string;
  resolvedAt?: string;
}

export interface AlertRule {
  ruleId: string;
  name: string;
  enabled: boolean;
  alertType: Alert['alertType'];
  condition: {
    metric: string;
    operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte';
    threshold: number;
  };
  severity: Alert['severity'];
  notifications: {
    type: 'webhook' | 'slack' | 'pagerduty' | 'email';
    config: Record<string, string>;
  }[];
}
