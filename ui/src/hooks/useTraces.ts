/**
 * Traces data fetching hooks
 */

import { useState, useEffect, useCallback } from 'react';
import { tracesApi, Trace, TraceFilters, PaginationParams, TimeRangeParams } from '../services/api';

export interface UseTracesOptions extends PaginationParams, TimeRangeParams, TraceFilters {}

export interface UseTracesResult {
  traces: Trace[];
  loading: boolean;
  error: string | null;
  pagination: { page: number; pageSize: number; total: number };
  refetch: () => void;
}

export function useTraces(options: UseTracesOptions = {}): UseTracesResult {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState({ page: 1, pageSize: 50, total: 0 });

  const fetchTraces = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await tracesApi.list(options);
      setTraces(response.data);
      if (response.pagination) {
        setPagination(response.pagination);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch traces');
    } finally {
      setLoading(false);
    }
  }, [JSON.stringify(options)]);

  useEffect(() => {
    fetchTraces();
  }, [fetchTraces]);

  return { traces, loading, error, pagination, refetch: fetchTraces };
}

export interface UseTraceResult {
  trace: Trace | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useTrace(traceId: string): UseTraceResult {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTrace = useCallback(async () => {
    if (!traceId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await tracesApi.get(traceId);
      setTrace(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trace');
    } finally {
      setLoading(false);
    }
  }, [traceId]);

  useEffect(() => {
    fetchTrace();
  }, [fetchTrace]);

  return { trace, loading, error, refetch: fetchTrace };
}

export interface TraceStats {
  totalTraces: number;
  errorRate: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  totalCostMicros: number;
  tracesByModel: Record<string, number>;
}

export function useTraceStats(timeRange: TimeRangeParams = {}) {
  const [stats, setStats] = useState<TraceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await tracesApi.getStats(timeRange);
      setStats(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stats');
    } finally {
      setLoading(false);
    }
  }, [JSON.stringify(timeRange)]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return { stats, loading, error, refetch: fetchStats };
}

export function useTraceTimeline(traceId: string) {
  const [timeline, setTimeline] = useState<{ traceId: string; spans: Trace[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!traceId) return;

    const fetchTimeline = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await tracesApi.getTimeline(traceId);
        setTimeline(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch timeline');
      } finally {
        setLoading(false);
      }
    };

    fetchTimeline();
  }, [traceId]);

  return { timeline, loading, error };
}
