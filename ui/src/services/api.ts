/**
 * PyFlare API Service
 * Centralized API client with type-safe endpoints
 *
 * Security Features:
 * - CSRF protection on all state-changing requests
 * - Credentials included for httpOnly cookie auth
 * - Structured query parameters (no raw SQL)
 * - Input validation helpers
 */

import { getCSRFHeaders } from '../utils/csrf';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

export interface PaginationParams {
  page?: number;
  pageSize?: number;
}

export interface TimeRangeParams {
  start?: string;
  end?: string;
  interval?: string;
}

export interface ApiResponse<T> {
  data: T;
  pagination?: {
    page: number;
    pageSize: number;
    total: number;
  };
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Sanitize string input to prevent injection attacks
 */
function sanitizeString(value: string | undefined): string | undefined {
  if (!value) return value;
  // Remove potential injection characters and limit length
  return value
    .replace(/[<>'"`;\\]/g, '')
    .slice(0, 1000);
}

/**
 * Validate and sanitize ID parameters
 */
function validateId(id: string): string {
  // IDs should be alphanumeric with dashes/underscores
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
    throw new ApiError(400, 'Invalid ID format');
  }
  return id;
}

/**
 * Build URL search params with sanitization
 */
function buildParams(params: Record<string, unknown>): URLSearchParams {
  const sanitized: Record<string, string> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      if (typeof value === 'string') {
        const sanitizedValue = sanitizeString(value);
        if (sanitizedValue) {
          sanitized[key] = sanitizedValue;
        }
      } else if (typeof value === 'number' || typeof value === 'boolean') {
        sanitized[key] = String(value);
      }
    }
  }
  return new URLSearchParams(sanitized);
}

/**
 * Make a GET request
 */
async function get<T>(endpoint: string, params: Record<string, unknown> = {}): Promise<T> {
  const searchParams = buildParams(params);
  const url = `${API_BASE}${endpoint}${searchParams.toString() ? `?${searchParams}` : ''}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...getCSRFHeaders(),
    },
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new ApiError(response.status, error.error || error.message);
  }

  return response.json();
}

/**
 * Make a POST request with CSRF protection
 */
async function post<T>(endpoint: string, data?: unknown): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getCSRFHeaders(),
    },
    credentials: 'include',
    body: data ? JSON.stringify(data) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new ApiError(response.status, error.error || error.message);
  }

  return response.json();
}

/**
 * Make a PUT request with CSRF protection
 */
async function put<T>(endpoint: string, data?: unknown): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...getCSRFHeaders(),
    },
    credentials: 'include',
    body: data ? JSON.stringify(data) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new ApiError(response.status, error.error || error.message);
  }

  return response.json();
}

/**
 * Make a DELETE request with CSRF protection
 */
async function del<T>(endpoint: string): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
      ...getCSRFHeaders(),
    },
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new ApiError(response.status, error.error || error.message);
  }

  return response.json();
}

// =============================================================================
// Traces API
// =============================================================================

export interface Trace {
  traceId: string;
  spanId: string;
  modelId: string;
  userId: string;
  featureId: string;
  startTime: string;
  endTime: string;
  latencyMs: number;
  status: 'ok' | 'error';
  errorType?: string;
  errorMessage?: string;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  costMicros: number;
  evalScore?: number;
  driftScore?: number;
  toxicityScore?: number;
  input?: string;
  output?: string;
}

export interface TraceFilters {
  modelId?: string;
  userId?: string;
  featureId?: string;
  status?: 'ok' | 'error';
  minLatencyMs?: number;
  maxLatencyMs?: number;
  minCostMicros?: number;
  maxCostMicros?: number;
  hasError?: boolean;
  hasDrift?: boolean;
}

/**
 * Structured search query (no raw SQL - SEC-006 fix)
 */
export interface TraceSearchQuery {
  filters?: TraceFilters;
  textSearch?: string;
  sortBy?: 'timestamp' | 'latency' | 'cost' | 'tokens';
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export const tracesApi = {
  list: (params: PaginationParams & TimeRangeParams & TraceFilters = {}) =>
    get<ApiResponse<Trace[]>>('/traces', params),

  get: (traceId: string) =>
    get<Trace>(`/traces/${validateId(traceId)}`),

  /**
   * Search traces with structured filters (no raw SQL)
   * Removed sql parameter to prevent injection attacks
   */
  search: (query: TraceSearchQuery) =>
    post<ApiResponse<Trace[]>>('/traces/search', {
      filters: query.filters,
      textSearch: sanitizeString(query.textSearch),
      sortBy: query.sortBy,
      sortOrder: query.sortOrder,
      limit: Math.min(query.limit || 100, 1000), // Cap at 1000
      offset: query.offset || 0,
    }),

  getStats: (params: TimeRangeParams = {}) =>
    get<{
      totalTraces: number;
      errorRate: number;
      avgLatencyMs: number;
      p95LatencyMs: number;
      totalCostMicros: number;
      tracesByModel: Record<string, number>;
    }>('/traces/stats', params),

  getTimeline: (traceId: string) =>
    get<{ traceId: string; spans: Trace[] }>(`/traces/${validateId(traceId)}/timeline`),
};

// =============================================================================
// Drift API
// =============================================================================

export interface DriftAlert {
  id: string;
  modelId: string;
  driftType: 'feature' | 'embedding' | 'concept';
  severity: 'low' | 'medium' | 'high' | 'critical';
  score: number;
  threshold: number;
  featureName?: string;
  timestamp: string;
  resolvedAt?: string;
  isResolved: boolean;
  description: string;
}

export interface DriftStatus {
  modelId: string;
  status: 'healthy' | 'drifted';
  featureDriftScore: number;
  embeddingDriftScore: number;
  lastChecked: string;
  activeAlerts: number;
}

export const driftApi = {
  listAlerts: (params: PaginationParams & { severity?: string; isResolved?: boolean } = {}) =>
    get<ApiResponse<DriftAlert[]>>('/drift/alerts', params),

  getStatus: (modelId: string) =>
    get<DriftStatus>(`/drift/models/${validateId(modelId)}`),

  getTimeline: (params: TimeRangeParams & { modelId?: string } = {}) =>
    get<{ timeline: { timestamp: string; score: number }[] }>('/drift/timeline', params),

  getHeatmap: () =>
    get<{ features: string[]; models: string[]; matrix: number[][] }>('/drift/heatmap'),

  updateReference: (data: { modelId: string; featureName?: string }) =>
    post<{ status: string }>('/drift/reference', {
      modelId: validateId(data.modelId),
      featureName: sanitizeString(data.featureName),
    }),

  resolveAlert: (alertId: string) =>
    post<{ alertId: string; resolved: boolean }>(`/drift/alerts/${validateId(alertId)}/resolve`),
};

// =============================================================================
// Costs API
// =============================================================================

export interface CostSummary {
  totalCostMicros: number;
  inputCostMicros: number;
  outputCostMicros: number;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  avgCostPerRequestMicros: number;
  period: { start: string; end: string };
}

export interface CostBreakdownItem {
  dimension: string;
  value: string;
  costMicros: number;
  tokens: number;
  requests: number;
  percentage: number;
}

export interface Budget {
  id: string;
  dimension: string;
  dimensionValue: string;
  period: 'hourly' | 'daily' | 'weekly' | 'monthly';
  softLimitMicros: number;
  hardLimitMicros: number;
  currentSpendMicros: number;
  utilizationPercentage: number;
  warningTriggered: boolean;
  limitExceeded: boolean;
}

export const costsApi = {
  getSummary: (params: TimeRangeParams = {}) =>
    get<CostSummary>('/costs/summary', params),

  getBreakdown: (params: TimeRangeParams & { dimension?: string } = {}) =>
    get<{ dimension: string; breakdown: CostBreakdownItem[] }>('/costs/breakdown', params),

  getTimeline: (params: TimeRangeParams & { granularity?: string } = {}) =>
    get<{ timeline: { timestamp: string; costMicros: number; tokens: number }[] }>(
      '/costs/timeline',
      params
    ),

  listBudgets: () =>
    get<{ budgets: Budget[] }>('/costs/budgets'),

  createBudget: (budget: Partial<Budget>) =>
    post<{ budgetId: string; created: boolean }>('/costs/budgets', budget),

  updateBudget: (budgetId: string, budget: Partial<Budget>) =>
    put<{ budgetId: string; updated: boolean }>(`/costs/budgets/${validateId(budgetId)}`, budget),

  deleteBudget: (budgetId: string) =>
    del<{ budgetId: string; deleted: boolean }>(`/costs/budgets/${validateId(budgetId)}`),

  getForecast: () =>
    get<{ forecast: { date: string; projectedCostMicros: number }[]; projectedTotalMicros: number }>(
      '/costs/forecast'
    ),
};

// =============================================================================
// Evaluations API
// =============================================================================

export interface Evaluation {
  id: string;
  traceId: string;
  modelId: string;
  evaluatorType: 'hallucination' | 'toxicity' | 'rag_quality';
  score: number;
  verdict: 'pass' | 'fail' | 'warn';
  explanation: string;
  timestamp: string;
  metadata: Record<string, string>;
}

export interface EvaluationSummary {
  totalEvaluations: number;
  passRate: number;
  failRate: number;
  warnRate: number;
  byType: Record<string, { total: number; passRate: number }>;
}

export const evaluationsApi = {
  list: (params: PaginationParams & { evaluatorType?: string; verdict?: string } = {}) =>
    get<ApiResponse<Evaluation[]>>('/evaluations', params),

  get: (evalId: string) =>
    get<Evaluation>(`/evaluations/${validateId(evalId)}`),

  getSummary: (params: TimeRangeParams = {}) =>
    get<EvaluationSummary>('/evaluations/summary', params),

  getByTrace: (traceId: string) =>
    get<{ traceId: string; evaluations: Evaluation[] }>(
      `/traces/${validateId(traceId)}/evaluations`
    ),

  trigger: (data: { traceIds?: string[]; evaluatorTypes?: string[] }) =>
    post<{ jobId: string; status: string }>('/evaluations', {
      traceIds: data.traceIds?.map(validateId),
      evaluatorTypes: data.evaluatorTypes,
    }),

  getTrends: (params: TimeRangeParams = {}) =>
    get<{ trends: { date: string; passRate: number; avgScore: number }[] }>(
      '/evaluations/trends',
      params
    ),
};

// =============================================================================
// RCA API
// =============================================================================

export interface RCAReport {
  id: string;
  modelId: string;
  analysisType: 'failure_analysis' | 'performance_analysis';
  timestamp: string;
  status: 'running' | 'completed' | 'failed';
  tracesAnalyzed: number;
  patternsFound: number;
  clustersFound: number;
  patterns: Pattern[];
  clusters: FailureCluster[];
  recommendations: string[];
}

export interface Pattern {
  id: string;
  type: string;
  description: string;
  affectedTraces: number;
  severity: number;
  suggestedActions: string[];
}

export interface FailureCluster {
  id: string;
  name: string;
  size: number;
  representativeError: string;
  commonKeywords: string[];
  severity: number;
}

export interface ProblematicSlice {
  id: string;
  name: string;
  dimension: string;
  dimensionValue: string;
  metric: string;
  metricValue: number;
  baseline: number;
  deviation: number;
  deviationPercentage: number;
  isSignificant: boolean;
  impactScore: number;
}

export const rcaApi = {
  runAnalysis: (data: { modelId?: string; traceIds?: string[]; analysisType?: string }) =>
    post<{ reportId: string; status: string }>('/rca/analyze', {
      modelId: data.modelId ? validateId(data.modelId) : undefined,
      traceIds: data.traceIds?.map(validateId),
      analysisType: data.analysisType,
    }),

  getReport: (reportId: string) =>
    get<RCAReport>(`/rca/reports/${validateId(reportId)}`),

  listReports: (params: PaginationParams = {}) =>
    get<ApiResponse<RCAReport[]>>('/rca/reports', params),

  getPatterns: (params: TimeRangeParams & { modelId?: string } = {}) =>
    get<{ patterns: Pattern[] }>('/rca/patterns', params),

  getClusters: (params: TimeRangeParams & { modelId?: string } = {}) =>
    get<{ clusters: FailureCluster[] }>('/rca/clusters', params),

  getSlices: (params: { modelId?: string; metric?: string } = {}) =>
    get<{ slices: ProblematicSlice[] }>('/rca/slices', params),

  getSliceDetail: (sliceId: string) =>
    get<ProblematicSlice>(`/rca/slices/${validateId(sliceId)}`),
};

export default {
  traces: tracesApi,
  drift: driftApi,
  costs: costsApi,
  evaluations: evaluationsApi,
  rca: rcaApi,
};
