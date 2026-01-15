/**
 * Drift detection data fetching hooks
 */

import { useState, useEffect, useCallback } from 'react';
import { driftApi, DriftAlert, DriftStatus, TimeRangeParams } from '../services/api';

export function useDriftAlerts(filters: { severity?: string; isResolved?: boolean } = {}) {
  const [alerts, setAlerts] = useState<DriftAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await driftApi.listAlerts(filters);
      setAlerts(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch alerts');
    } finally {
      setLoading(false);
    }
  }, [JSON.stringify(filters)]);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  const resolveAlert = async (alertId: string) => {
    try {
      await driftApi.resolveAlert(alertId);
      fetchAlerts();
    } catch (err) {
      throw err;
    }
  };

  return { alerts, loading, error, refetch: fetchAlerts, resolveAlert };
}

export function useDriftStatus(modelId: string) {
  const [status, setStatus] = useState<DriftStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!modelId) return;

    const fetchStatus = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await driftApi.getStatus(modelId);
        setStatus(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch status');
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, [modelId]);

  return { status, loading, error };
}

export function useDriftTimeline(params: TimeRangeParams & { modelId?: string } = {}) {
  const [timeline, setTimeline] = useState<{ timestamp: string; score: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTimeline = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await driftApi.getTimeline(params);
        setTimeline(response.timeline);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch timeline');
      } finally {
        setLoading(false);
      }
    };

    fetchTimeline();
  }, [JSON.stringify(params)]);

  return { timeline, loading, error };
}

export function useDriftHeatmap() {
  const [heatmap, setHeatmap] = useState<{
    features: string[];
    models: string[];
    matrix: number[][];
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHeatmap = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await driftApi.getHeatmap();
        setHeatmap(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch heatmap');
      } finally {
        setLoading(false);
      }
    };

    fetchHeatmap();
  }, []);

  return { heatmap, loading, error };
}
