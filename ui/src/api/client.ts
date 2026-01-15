/**
 * API Client for PyFlare
 *
 * SECURITY: This client uses secure token storage and CSRF protection.
 * Tokens are stored in encrypted sessionStorage (not localStorage) to reduce
 * XSS attack surface. CSRF tokens are included in all state-changing requests.
 */
import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type { Trace, Span, DriftStatus, CostSummary, Alert } from '@/types';
import { secureStorage, isSecureStorageAvailable } from '@/utils/secureStorage';
import { getCSRFHeaders } from '@/utils/csrf';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const TOKEN_STORAGE_KEY = 'auth_token';
const TOKEN_TTL_MS = 15 * 60 * 1000; // 15 minutes

// In-memory token cache for synchronous access in interceptors
let cachedToken: string | null = null;

class ApiClient {
  private client: AxiosInstance;
  private tokenPromise: Promise<string | null> | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      withCredentials: true, // Enable httpOnly cookie support
    });

    // Initialize token from secure storage
    this.initializeToken();

    // Add request interceptor for auth and CSRF
    this.client.interceptors.request.use(
      async (config: InternalAxiosRequestConfig) => {
        // Add CSRF headers to all requests
        const csrfHeaders = getCSRFHeaders();
        Object.assign(config.headers, csrfHeaders);

        // Add auth token if available
        const token = await this.getToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized - clear token securely
          await this.clearToken();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  /**
   * Initialize token from secure storage on startup
   */
  private async initializeToken(): Promise<void> {
    try {
      const token = await this.getToken();
      cachedToken = token;
    } catch {
      cachedToken = null;
    }
  }

  /**
   * Get auth token from secure storage
   * Uses caching to avoid repeated decryption
   */
  private async getToken(): Promise<string | null> {
    // Return cached token if available
    if (cachedToken) {
      return cachedToken;
    }

    // Deduplicate concurrent token fetch requests
    if (this.tokenPromise) {
      return this.tokenPromise;
    }

    this.tokenPromise = (async () => {
      try {
        if (isSecureStorageAvailable()) {
          const token = await secureStorage.getItem(TOKEN_STORAGE_KEY);
          cachedToken = token;
          return token;
        }
        return null;
      } finally {
        this.tokenPromise = null;
      }
    })();

    return this.tokenPromise;
  }

  /**
   * Store auth token securely
   */
  async setToken(token: string): Promise<void> {
    cachedToken = token;
    if (isSecureStorageAvailable()) {
      await secureStorage.setItem(TOKEN_STORAGE_KEY, token, TOKEN_TTL_MS);
    }
  }

  /**
   * Clear auth token securely
   */
  async clearToken(): Promise<void> {
    cachedToken = null;
    if (isSecureStorageAvailable()) {
      secureStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  }

  // Traces
  async getTraces(params?: {
    service?: string;
    modelId?: string;
    status?: 'ok' | 'error';
    startTime?: string;
    endTime?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ traces: Trace[]; total: number }> {
    const response = await this.client.get('/traces', { params });
    return response.data;
  }

  async getTrace(traceId: string): Promise<Trace> {
    const response = await this.client.get(`/traces/${traceId}`);
    return response.data;
  }

  async getTraceSpans(traceId: string): Promise<Span[]> {
    const response = await this.client.get(`/traces/${traceId}/spans`);
    return response.data;
  }

  // Drift
  async getDriftStatus(modelId: string): Promise<DriftStatus> {
    const response = await this.client.get(`/drift/${modelId}`);
    return response.data;
  }

  async getDriftHistory(modelId: string, params?: {
    driftType?: 'feature' | 'embedding' | 'concept' | 'prediction';
    startTime?: string;
    endTime?: string;
  }): Promise<{ history: DriftStatus[] }> {
    const response = await this.client.get(`/drift/${modelId}/history`, { params });
    return response.data;
  }

  // Costs
  async getCostSummary(params: {
    startTime: string;
    endTime: string;
    groupBy?: string[];
  }): Promise<CostSummary> {
    const response = await this.client.get('/costs', { params });
    return response.data;
  }

  async getCostBreakdown(params: {
    dimension: 'model' | 'user' | 'feature' | 'team';
    startTime: string;
    endTime: string;
    limit?: number;
  }): Promise<{ breakdown: { name: string; costUsd: number; requests: number }[] }> {
    const response = await this.client.get('/costs/breakdown', { params });
    return response.data;
  }

  // Alerts
  async getAlerts(params?: {
    status?: Alert['status'];
    severity?: Alert['severity'];
    limit?: number;
  }): Promise<{ alerts: Alert[] }> {
    const response = await this.client.get('/alerts', { params });
    return response.data;
  }

  async acknowledgeAlert(alertId: string): Promise<void> {
    await this.client.post(`/alerts/${alertId}/acknowledge`);
  }

  // Query
  async executeQuery(sql: string, params?: Record<string, unknown>): Promise<{
    columns: string[];
    rows: unknown[][];
    totalRows: number;
    executionTimeMs: number;
  }> {
    const response = await this.client.post('/query', { sql, params });
    return response.data;
  }
}

export const api = new ApiClient();
export default api;
