import { DataSourceJsonData, DataQuery } from '@grafana/data';

/**
 * Query types supported by the PyFlare data source
 */
export type QueryType = 'traces' | 'metrics' | 'drift' | 'costs' | 'evaluations';

/**
 * PyFlare query definition
 */
export interface PyFlareQuery extends DataQuery {
  queryType: QueryType;

  // Trace query options
  traceId?: string;
  serviceName?: string;
  modelId?: string;
  status?: 'ok' | 'error' | 'all';
  minDuration?: number;
  maxDuration?: number;

  // Metric query options
  metric?: string;
  aggregation?: 'avg' | 'sum' | 'min' | 'max' | 'count' | 'p50' | 'p95' | 'p99';
  groupBy?: string[];

  // Drift query options
  driftType?: 'feature' | 'embedding' | 'concept' | 'prediction' | 'all';
  threshold?: number;

  // Cost query options
  costGroupBy?: 'model' | 'service' | 'user' | 'feature';

  // Evaluation query options
  evalMetric?: 'accuracy' | 'latency' | 'toxicity' | 'hallucination' | 'relevance';

  // Common filters
  timeRange?: string;
  limit?: number;
}

/**
 * Default query values
 */
export const defaultQuery: Partial<PyFlareQuery> = {
  queryType: 'metrics',
  metric: 'request_count',
  aggregation: 'sum',
  limit: 100,
};

/**
 * Data source configuration options
 */
export interface PyFlareDataSourceOptions extends DataSourceJsonData {
  url: string;
  defaultServiceName?: string;
  defaultModelId?: string;
  timeout?: number;
}

/**
 * Secure configuration options (stored encrypted)
 */
export interface PyFlareSecureJsonData {
  apiKey?: string;
}

/**
 * Available metrics for querying
 */
export const AVAILABLE_METRICS = [
  { value: 'request_count', label: 'Request Count' },
  { value: 'latency_ms', label: 'Latency (ms)' },
  { value: 'tokens_input', label: 'Input Tokens' },
  { value: 'tokens_output', label: 'Output Tokens' },
  { value: 'tokens_total', label: 'Total Tokens' },
  { value: 'cost_micros', label: 'Cost (micros)' },
  { value: 'error_rate', label: 'Error Rate' },
  { value: 'drift_score', label: 'Drift Score' },
  { value: 'eval_score', label: 'Evaluation Score' },
  { value: 'toxicity_score', label: 'Toxicity Score' },
];

/**
 * Aggregation options
 */
export const AGGREGATION_OPTIONS = [
  { value: 'avg', label: 'Average' },
  { value: 'sum', label: 'Sum' },
  { value: 'min', label: 'Minimum' },
  { value: 'max', label: 'Maximum' },
  { value: 'count', label: 'Count' },
  { value: 'p50', label: 'P50 (Median)' },
  { value: 'p95', label: 'P95' },
  { value: 'p99', label: 'P99' },
];

/**
 * Group by options
 */
export const GROUP_BY_OPTIONS = [
  { value: 'model_id', label: 'Model' },
  { value: 'service_name', label: 'Service' },
  { value: 'user_id', label: 'User' },
  { value: 'status', label: 'Status' },
  { value: 'inference_type', label: 'Inference Type' },
];

/**
 * Drift type options
 */
export const DRIFT_TYPE_OPTIONS = [
  { value: 'all', label: 'All Types' },
  { value: 'feature', label: 'Feature Drift' },
  { value: 'embedding', label: 'Embedding Drift' },
  { value: 'concept', label: 'Concept Drift' },
  { value: 'prediction', label: 'Prediction Drift' },
];
