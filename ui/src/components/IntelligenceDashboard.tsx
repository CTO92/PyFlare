import React, { useState, useEffect, useCallback } from 'react';

// Types
interface ModelHealth {
  model_id: string;
  health_score: number;
  has_active_drift: boolean;
  active_alerts: number;
  recent_safety_issues: number;
  avg_evaluation_score: number;
  last_analyzed: number;
}

interface SystemHealth {
  overall_health: number;
  models_with_drift: number;
  total_active_alerts: number;
  models_analyzed: number;
  avg_health_score: number;
  last_update: number;
}

interface PipelineStats {
  total_processed: number;
  drift_detections: number;
  safety_issues: number;
  evaluation_failures: number;
  rca_triggered: number;
  alerts_generated: number;
  avg_processing_time_ms: number;
  p99_processing_time_ms: number;
  queue_depth: number;
  component_health: {
    drift_service: boolean;
    eval_service: boolean;
    rca_service: boolean;
    alert_service: boolean;
  };
}

interface IntelligenceResult {
  trace_id: string;
  model_id: string;
  analyzed_at: number;
  health_score: number;
  drift: {
    drift_detected: boolean;
    overall_severity: number;
    drifted_dimensions: string[];
    causes: string[];
  };
  evaluation: {
    overall_score: number;
    passed: boolean;
    issues: string[];
  };
  safety: {
    is_safe: boolean;
    risk_score: number;
    detected_issues: string[];
    risk_level: string;
  };
}

// API functions
const API_BASE = '/api/v1/intelligence';

async function fetchSystemHealth(): Promise<SystemHealth> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Failed to fetch system health');
  return response.json();
}

async function fetchModels(): Promise<ModelHealth[]> {
  const response = await fetch(`${API_BASE}/models`);
  if (!response.ok) throw new Error('Failed to fetch models');
  return response.json();
}

async function fetchStats(): Promise<PipelineStats> {
  const response = await fetch(`${API_BASE}/stats`);
  if (!response.ok) throw new Error('Failed to fetch stats');
  return response.json();
}

async function fetchModelHealth(modelId: string): Promise<ModelHealth> {
  const response = await fetch(`${API_BASE}/health/${modelId}`);
  if (!response.ok) throw new Error('Failed to fetch model health');
  return response.json();
}

// Helper components
const HealthIndicator: React.FC<{ score: number }> = ({ score }) => {
  const getColor = () => {
    if (score >= 0.8) return 'text-green-500';
    if (score >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getLabel = () => {
    if (score >= 0.8) return 'Healthy';
    if (score >= 0.6) return 'Warning';
    return 'Critical';
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${getColor().replace('text-', 'bg-')}`} />
      <span className={getColor()}>{getLabel()}</span>
      <span className="text-gray-500">({(score * 100).toFixed(0)}%)</span>
    </div>
  );
};

const StatCard: React.FC<{
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  alert?: boolean;
}> = ({ title, value, subtitle, trend, alert }) => {
  return (
    <div className={`p-4 rounded-lg border ${alert ? 'border-red-300 bg-red-50' : 'border-gray-200 bg-white'}`}>
      <div className="text-sm text-gray-500">{title}</div>
      <div className={`text-2xl font-bold ${alert ? 'text-red-600' : 'text-gray-900'}`}>
        {value}
        {trend && (
          <span className="ml-2 text-sm">
            {trend === 'up' && '↑'}
            {trend === 'down' && '↓'}
          </span>
        )}
      </div>
      {subtitle && <div className="text-xs text-gray-400">{subtitle}</div>}
    </div>
  );
};

const ComponentStatus: React.FC<{ name: string; healthy: boolean }> = ({ name, healthy }) => {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-gray-600">{name}</span>
      <span className={`px-2 py-1 rounded text-xs ${healthy ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
        {healthy ? 'Healthy' : 'Unhealthy'}
      </span>
    </div>
  );
};

const ModelHealthRow: React.FC<{ model: ModelHealth; onClick: () => void }> = ({ model, onClick }) => {
  return (
    <tr
      className="hover:bg-gray-50 cursor-pointer border-b border-gray-100"
      onClick={onClick}
    >
      <td className="py-3 px-4">
        <span className="font-medium text-gray-900">{model.model_id}</span>
      </td>
      <td className="py-3 px-4">
        <HealthIndicator score={model.health_score} />
      </td>
      <td className="py-3 px-4">
        {model.has_active_drift ? (
          <span className="px-2 py-1 rounded bg-yellow-100 text-yellow-800 text-xs">Drift</span>
        ) : (
          <span className="px-2 py-1 rounded bg-green-100 text-green-800 text-xs">Stable</span>
        )}
      </td>
      <td className="py-3 px-4">
        {model.active_alerts > 0 ? (
          <span className="px-2 py-1 rounded bg-red-100 text-red-800 text-xs">
            {model.active_alerts} alerts
          </span>
        ) : (
          <span className="text-gray-400">-</span>
        )}
      </td>
      <td className="py-3 px-4 text-gray-500 text-sm">
        {model.last_analyzed
          ? new Date(model.last_analyzed * 1000).toLocaleString()
          : 'Never'}
      </td>
    </tr>
  );
};

// Main component
const IntelligenceDashboard: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [models, setModels] = useState<ModelHealth[]>([]);
  const [stats, setStats] = useState<PipelineStats | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [health, modelList, pipelineStats] = await Promise.all([
        fetchSystemHealth(),
        fetchModels(),
        fetchStats(),
      ]);
      setSystemHealth(health);
      setModels(modelList);
      setStats(pipelineStats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [loadData]);

  if (loading && !systemHealth) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading intelligence data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="text-red-600 font-medium">Error loading data</div>
        <div className="text-red-500 text-sm">{error}</div>
        <button
          onClick={loadData}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Intelligence Dashboard</h1>
        <button
          onClick={loadData}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
        >
          Refresh
        </button>
      </div>

      {/* System Health Overview */}
      {systemHealth && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard
            title="Overall Health"
            value={`${(systemHealth.overall_health * 100).toFixed(0)}%`}
            alert={systemHealth.overall_health < 0.6}
          />
          <StatCard
            title="Models Analyzed"
            value={systemHealth.models_analyzed}
          />
          <StatCard
            title="Active Alerts"
            value={systemHealth.total_active_alerts}
            alert={systemHealth.total_active_alerts > 0}
          />
          <StatCard
            title="Models with Drift"
            value={systemHealth.models_with_drift}
            alert={systemHealth.models_with_drift > 0}
          />
        </div>
      )}

      {/* Pipeline Stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Processing Stats */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Processing Statistics</h2>
            <div className="grid grid-cols-2 gap-4">
              <StatCard title="Total Processed" value={stats.total_processed.toLocaleString()} />
              <StatCard title="Drift Detections" value={stats.drift_detections} alert={stats.drift_detections > 0} />
              <StatCard title="Safety Issues" value={stats.safety_issues} alert={stats.safety_issues > 0} />
              <StatCard title="Eval Failures" value={stats.evaluation_failures} alert={stats.evaluation_failures > 0} />
            </div>
            <div className="mt-4 pt-4 border-t border-gray-100">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Avg Processing Time:</span>
                  <span className="ml-2 font-medium">{stats.avg_processing_time_ms.toFixed(2)}ms</span>
                </div>
                <div>
                  <span className="text-gray-500">P99 Latency:</span>
                  <span className="ml-2 font-medium">{stats.p99_processing_time_ms.toFixed(2)}ms</span>
                </div>
              </div>
            </div>
          </div>

          {/* Component Health */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Component Health</h2>
            <div className="space-y-2">
              <ComponentStatus name="Drift Service" healthy={stats.component_health.drift_service} />
              <ComponentStatus name="Evaluation Service" healthy={stats.component_health.eval_service} />
              <ComponentStatus name="RCA Service" healthy={stats.component_health.rca_service} />
              <ComponentStatus name="Alert Service" healthy={stats.component_health.alert_service} />
            </div>
            <div className="mt-4 pt-4 border-t border-gray-100">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Queue Depth:</span>
                <span className={`font-medium ${stats.queue_depth > 100 ? 'text-yellow-600' : 'text-gray-900'}`}>
                  {stats.queue_depth}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Health Table */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Model Health</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Model ID</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Health Score</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Drift Status</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Alerts</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Last Analyzed</th>
              </tr>
            </thead>
            <tbody>
              {models.length > 0 ? (
                models.map((model) => (
                  <ModelHealthRow
                    key={model.model_id}
                    model={model}
                    onClick={() => setSelectedModel(model.model_id)}
                  />
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-gray-500">
                    No models registered yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="flex gap-4">
          <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">
            Register Model
          </button>
          <button className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 text-sm">
            Configure Alerts
          </button>
          <button className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm">
            Run Manual RCA
          </button>
        </div>
      </div>
    </div>
  );
};

export default IntelligenceDashboard;
