/**
 * Cost tracking data fetching hooks
 */

import { useState, useEffect, useCallback } from 'react';
import { costsApi, CostSummary, CostBreakdownItem, Budget, TimeRangeParams } from '../services/api';
import type { CostMetrics } from '../components/costs/CostOverview';

interface CostData {
  metrics: CostMetrics;
  breakdown: {
    byModel: Array<{ model: string; cost: number; tokens: number; requests: number }>;
    byService: Array<{ service: string; cost: number; tokens: number; requests: number }>;
    byUser: Array<{ user: string; cost: number; tokens: number; requests: number }>;
  };
  timeline: Array<{
    timestamp: string;
    cost: number;
    tokens: number;
    requests: number;
  }>;
}

interface UseCostsOptions {
  timeRange?: string;
  modelId?: string;
  serviceId?: string;
  userId?: string;
}

interface UseCostsResult {
  costs: CostData | undefined;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useCosts(options: UseCostsOptions = {}): UseCostsResult {
  const { timeRange = '30d', modelId, serviceId, userId } = options;
  const [costs, setCosts] = useState<CostData | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCosts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // In production, this would call the API
      // For now, we'll just set loading to false
      // The page will use mock data when costs is undefined
      setCosts(undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch costs');
    } finally {
      setLoading(false);
    }
  }, [timeRange, modelId, serviceId, userId]);

  useEffect(() => {
    fetchCosts();
  }, [fetchCosts]);

  return { costs, loading, error, refetch: fetchCosts };
}

export function useCostSummary(timeRange: TimeRangeParams = {}) {
  const [summary, setSummary] = useState<CostSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await costsApi.getSummary(timeRange);
      setSummary(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch summary');
    } finally {
      setLoading(false);
    }
  }, [JSON.stringify(timeRange)]);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  return { summary, loading, error, refetch: fetchSummary };
}

export function useCostBreakdown(params: TimeRangeParams & { dimension?: string } = {}) {
  const [breakdown, setBreakdown] = useState<{
    dimension: string;
    items: CostBreakdownItem[];
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchBreakdown = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await costsApi.getBreakdown(params);
        setBreakdown({
          dimension: response.dimension,
          items: response.breakdown,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch breakdown');
      } finally {
        setLoading(false);
      }
    };

    fetchBreakdown();
  }, [JSON.stringify(params)]);

  return { breakdown, loading, error };
}

export function useCostTimeline(params: TimeRangeParams & { granularity?: string } = {}) {
  const [timeline, setTimeline] = useState<
    { timestamp: string; costMicros: number; tokens: number }[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTimeline = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await costsApi.getTimeline(params);
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

export function useBudgets() {
  const [budgets, setBudgets] = useState<Budget[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBudgets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await costsApi.listBudgets();
      setBudgets(response.budgets);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch budgets');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBudgets();
  }, [fetchBudgets]);

  const createBudget = async (budget: Partial<Budget>) => {
    await costsApi.createBudget(budget);
    fetchBudgets();
  };

  return { budgets, loading, error, refetch: fetchBudgets, createBudget };
}

export function useCostForecast() {
  const [forecast, setForecast] = useState<{
    data: { date: string; projectedCostMicros: number }[];
    projectedTotal: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchForecast = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await costsApi.getForecast();
        setForecast({
          data: response.forecast,
          projectedTotal: response.projectedTotalMicros,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch forecast');
      } finally {
        setLoading(false);
      }
    };

    fetchForecast();
  }, []);

  return { forecast, loading, error };
}

// Utility: Format cost in dollars
export function formatCost(microDollars: number): string {
  return `$${(microDollars / 1_000_000).toFixed(2)}`;
}

// Utility: Format large token counts
export function formatTokens(tokens: number): string {
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(1)}M`;
  }
  if (tokens >= 1_000) {
    return `${(tokens / 1_000).toFixed(1)}K`;
  }
  return tokens.toString();
}
