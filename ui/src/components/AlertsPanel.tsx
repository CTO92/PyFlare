import React, { useState, useEffect, useCallback } from 'react';

// Types
interface AlertEvent {
  alert_id: string;
  rule_id: string;
  rule_name: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  triggered_at: number;
  is_firing: boolean;
  is_resolved: boolean;
  title: string;
  description: string;
  metric_name: string;
  metric_value: number;
  threshold_value: number;
  model_id: string;
  labels: Record<string, string>;
  fingerprint: string;
}

interface AlertGroup {
  group_id: string;
  group_key: string;
  total_count: number;
  max_severity: string;
  is_firing: boolean;
  alert_count: number;
}

interface AlertRule {
  rule_id: string;
  name: string;
  description: string;
  type: string;
  severity: string;
  enabled: boolean;
}

interface Silence {
  silence_id: string;
  created_by: string;
  comment: string;
  matchers: Record<string, string>;
  starts_at: number;
  ends_at: number;
  is_active: boolean;
}

interface AlertsStats {
  active_alerts: number;
  active_rules: number;
  active_silences: number;
  alerts_generated: number;
  notifications_sent: number;
  notifications_failed: number;
}

// API functions
const API_BASE = '/api/v1/alerts';

async function fetchAlerts(): Promise<AlertEvent[]> {
  const response = await fetch(API_BASE);
  if (!response.ok) throw new Error('Failed to fetch alerts');
  return response.json();
}

async function fetchGroups(): Promise<AlertGroup[]> {
  const response = await fetch(`${API_BASE}/groups`);
  if (!response.ok) throw new Error('Failed to fetch groups');
  return response.json();
}

async function fetchRules(): Promise<AlertRule[]> {
  const response = await fetch(`${API_BASE}/rules`);
  if (!response.ok) throw new Error('Failed to fetch rules');
  return response.json();
}

async function fetchSilences(): Promise<Silence[]> {
  const response = await fetch(`${API_BASE}/silences`);
  if (!response.ok) throw new Error('Failed to fetch silences');
  return response.json();
}

async function fetchStats(): Promise<AlertsStats> {
  const response = await fetch(`${API_BASE}/stats`);
  if (!response.ok) throw new Error('Failed to fetch stats');
  return response.json();
}

async function resolveAlert(fingerprint: string): Promise<void> {
  const response = await fetch(`${API_BASE}/${fingerprint}/resolve`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to resolve alert');
}

async function acknowledgeAlert(alertId: string, acknowledgedBy: string): Promise<void> {
  const response = await fetch(`${API_BASE}/${alertId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ acknowledged_by: acknowledgedBy }),
  });
  if (!response.ok) throw new Error('Failed to acknowledge alert');
}

async function createSilence(silence: Partial<Silence>): Promise<void> {
  const response = await fetch(`${API_BASE}/silences`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(silence),
  });
  if (!response.ok) throw new Error('Failed to create silence');
}

// Helper components
const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
  const colors: Record<string, string> = {
    info: 'bg-blue-100 text-blue-800',
    warning: 'bg-yellow-100 text-yellow-800',
    error: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800',
  };

  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${colors[severity] || colors.info}`}>
      {severity.toUpperCase()}
    </span>
  );
};

const AlertCard: React.FC<{
  alert: AlertEvent;
  onResolve: () => void;
  onAcknowledge: () => void;
  onSilence: () => void;
}> = ({ alert, onResolve, onAcknowledge, onSilence }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`border rounded-lg p-4 ${
      alert.severity === 'critical' ? 'border-red-300 bg-red-50' :
      alert.severity === 'error' ? 'border-orange-300 bg-orange-50' :
      alert.severity === 'warning' ? 'border-yellow-300 bg-yellow-50' :
      'border-gray-200 bg-white'
    }`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <SeverityBadge severity={alert.severity} />
            {alert.is_firing ? (
              <span className="px-2 py-1 rounded bg-red-100 text-red-800 text-xs">FIRING</span>
            ) : (
              <span className="px-2 py-1 rounded bg-green-100 text-green-800 text-xs">RESOLVED</span>
            )}
          </div>
          <h3 className="font-semibold text-gray-900">{alert.title}</h3>
          <p className="text-sm text-gray-600 mt-1">{alert.description}</p>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-gray-400 hover:text-gray-600"
        >
          {expanded ? '▼' : '▶'}
        </button>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Rule:</span>
              <span className="ml-2 text-gray-900">{alert.rule_name}</span>
            </div>
            <div>
              <span className="text-gray-500">Model:</span>
              <span className="ml-2 text-gray-900">{alert.model_id || 'N/A'}</span>
            </div>
            <div>
              <span className="text-gray-500">Metric:</span>
              <span className="ml-2 text-gray-900">{alert.metric_name}</span>
            </div>
            <div>
              <span className="text-gray-500">Value:</span>
              <span className="ml-2 text-gray-900">
                {alert.metric_value.toFixed(2)} (threshold: {alert.threshold_value.toFixed(2)})
              </span>
            </div>
            <div>
              <span className="text-gray-500">Triggered:</span>
              <span className="ml-2 text-gray-900">
                {new Date(alert.triggered_at * 1000).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Fingerprint:</span>
              <span className="ml-2 text-gray-900 font-mono text-xs">{alert.fingerprint}</span>
            </div>
          </div>

          {Object.keys(alert.labels).length > 0 && (
            <div className="mt-4">
              <span className="text-gray-500 text-sm">Labels:</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {Object.entries(alert.labels).map(([key, value]) => (
                  <span key={key} className="px-2 py-1 bg-gray-100 rounded text-xs">
                    {key}={value}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-2 mt-4">
            {alert.is_firing && (
              <>
                <button
                  onClick={onResolve}
                  className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                >
                  Resolve
                </button>
                <button
                  onClick={onAcknowledge}
                  className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                >
                  Acknowledge
                </button>
                <button
                  onClick={onSilence}
                  className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700"
                >
                  Silence
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const RuleRow: React.FC<{
  rule: AlertRule;
  onToggle: () => void;
  onEdit: () => void;
  onDelete: () => void;
}> = ({ rule, onToggle, onEdit, onDelete }) => {
  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50">
      <td className="py-3 px-4">
        <span className="font-medium text-gray-900">{rule.name}</span>
        <p className="text-xs text-gray-500">{rule.description}</p>
      </td>
      <td className="py-3 px-4">
        <span className="text-sm text-gray-600">{rule.type}</span>
      </td>
      <td className="py-3 px-4">
        <SeverityBadge severity={rule.severity} />
      </td>
      <td className="py-3 px-4">
        <button
          onClick={onToggle}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            rule.enabled ? 'bg-green-600' : 'bg-gray-300'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              rule.enabled ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </td>
      <td className="py-3 px-4">
        <div className="flex gap-2">
          <button
            onClick={onEdit}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            Edit
          </button>
          <button
            onClick={onDelete}
            className="text-red-600 hover:text-red-800 text-sm"
          >
            Delete
          </button>
        </div>
      </td>
    </tr>
  );
};

const SilenceCard: React.FC<{
  silence: Silence;
  onDelete: () => void;
}> = ({ silence, onDelete }) => {
  const isActive = silence.is_active &&
    Date.now() >= silence.starts_at * 1000 &&
    Date.now() < silence.ends_at * 1000;

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            {isActive ? (
              <span className="px-2 py-1 rounded bg-purple-100 text-purple-800 text-xs">ACTIVE</span>
            ) : (
              <span className="px-2 py-1 rounded bg-gray-100 text-gray-800 text-xs">INACTIVE</span>
            )}
          </div>
          <p className="text-sm text-gray-600">{silence.comment || 'No comment'}</p>
        </div>
        <button
          onClick={onDelete}
          className="text-red-600 hover:text-red-800 text-sm"
        >
          Delete
        </button>
      </div>
      <div className="mt-3 text-sm">
        <div className="text-gray-500">
          Created by: <span className="text-gray-900">{silence.created_by}</span>
        </div>
        <div className="text-gray-500">
          Expires: <span className="text-gray-900">
            {new Date(silence.ends_at * 1000).toLocaleString()}
          </span>
        </div>
        {Object.keys(silence.matchers).length > 0 && (
          <div className="mt-2">
            <span className="text-gray-500">Matchers:</span>
            <div className="flex flex-wrap gap-2 mt-1">
              {Object.entries(silence.matchers).map(([key, value]) => (
                <span key={key} className="px-2 py-1 bg-purple-50 rounded text-xs">
                  {key}={value}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Main component
type TabType = 'alerts' | 'rules' | 'silences';

const AlertsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('alerts');
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [groups, setGroups] = useState<AlertGroup[]>([]);
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [silences, setSilences] = useState<Silence[]>([]);
  const [stats, setStats] = useState<AlertsStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showSilenceModal, setShowSilenceModal] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<AlertEvent | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [alertsData, groupsData, rulesData, silencesData, statsData] = await Promise.all([
        fetchAlerts(),
        fetchGroups(),
        fetchRules(),
        fetchSilences(),
        fetchStats(),
      ]);
      setAlerts(alertsData);
      setGroups(groupsData);
      setRules(rulesData);
      setSilences(silencesData);
      setStats(statsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 15000); // Refresh every 15s
    return () => clearInterval(interval);
  }, [loadData]);

  const handleResolve = async (fingerprint: string) => {
    try {
      await resolveAlert(fingerprint);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resolve alert');
    }
  };

  const handleAcknowledge = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId, 'user');
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to acknowledge alert');
    }
  };

  const handleSilence = (alert: AlertEvent) => {
    setSelectedAlert(alert);
    setShowSilenceModal(true);
  };

  const handleCreateSilence = async (duration: number, comment: string) => {
    if (!selectedAlert) return;

    try {
      await createSilence({
        comment,
        created_by: 'user',
        matchers: { rule_id: selectedAlert.rule_id },
        duration_hours: duration,
      } as any);
      setShowSilenceModal(false);
      setSelectedAlert(null);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create silence');
    }
  };

  if (loading && alerts.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading alerts...</div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Alerts</h1>
        <button
          onClick={loadData}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
        >
          Refresh
        </button>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="p-4 bg-white border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-500">Active Alerts</div>
            <div className={`text-2xl font-bold ${stats.active_alerts > 0 ? 'text-red-600' : 'text-gray-900'}`}>
              {stats.active_alerts}
            </div>
          </div>
          <div className="p-4 bg-white border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-500">Active Rules</div>
            <div className="text-2xl font-bold text-gray-900">{stats.active_rules}</div>
          </div>
          <div className="p-4 bg-white border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-500">Active Silences</div>
            <div className="text-2xl font-bold text-gray-900">{stats.active_silences}</div>
          </div>
          <div className="p-4 bg-white border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-500">Notifications Sent</div>
            <div className="text-2xl font-bold text-gray-900">{stats.notifications_sent}</div>
          </div>
          <div className="p-4 bg-white border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-500">Notifications Failed</div>
            <div className={`text-2xl font-bold ${stats.notifications_failed > 0 ? 'text-red-600' : 'text-gray-900'}`}>
              {stats.notifications_failed}
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-8">
          {(['alerts', 'rules', 'silences'] as TabType[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
              {tab === 'alerts' && stats && stats.active_alerts > 0 && (
                <span className="ml-2 px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs">
                  {stats.active_alerts}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Error display */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="text-red-600">{error}</div>
        </div>
      )}

      {/* Tab content */}
      {activeTab === 'alerts' && (
        <div className="space-y-4">
          {alerts.length > 0 ? (
            alerts.map((alert) => (
              <AlertCard
                key={alert.alert_id}
                alert={alert}
                onResolve={() => handleResolve(alert.fingerprint)}
                onAcknowledge={() => handleAcknowledge(alert.alert_id)}
                onSilence={() => handleSilence(alert)}
              />
            ))
          ) : (
            <div className="text-center py-12 text-gray-500">
              No active alerts
            </div>
          )}
        </div>
      )}

      {activeTab === 'rules' && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-4 py-3 border-b border-gray-200 flex justify-between items-center">
            <h2 className="font-semibold text-gray-900">Alert Rules</h2>
            <button className="px-4 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
              Create Rule
            </button>
          </div>
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Name</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Type</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Severity</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Enabled</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((rule) => (
                <RuleRow
                  key={rule.rule_id}
                  rule={rule}
                  onToggle={() => {}}
                  onEdit={() => {}}
                  onDelete={() => {}}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'silences' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="font-semibold text-gray-900">Active Silences</h2>
            <button className="px-4 py-2 bg-purple-600 text-white rounded text-sm hover:bg-purple-700">
              Create Silence
            </button>
          </div>
          {silences.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {silences.map((silence) => (
                <SilenceCard
                  key={silence.silence_id}
                  silence={silence}
                  onDelete={() => {}}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              No active silences
            </div>
          )}
        </div>
      )}

      {/* Silence Modal */}
      {showSilenceModal && selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96">
            <h3 className="text-lg font-semibold mb-4">Create Silence</h3>
            <p className="text-sm text-gray-600 mb-4">
              Silence alerts from rule: <strong>{selectedAlert.rule_name}</strong>
            </p>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Duration
                </label>
                <select className="w-full border border-gray-300 rounded px-3 py-2">
                  <option value="1">1 hour</option>
                  <option value="2">2 hours</option>
                  <option value="4">4 hours</option>
                  <option value="8">8 hours</option>
                  <option value="24">24 hours</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Comment
                </label>
                <textarea
                  className="w-full border border-gray-300 rounded px-3 py-2"
                  rows={3}
                  placeholder="Reason for silencing..."
                />
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => setShowSilenceModal(false)}
                className="px-4 py-2 border border-gray-300 rounded text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleCreateSilence(1, '')}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                Create Silence
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertsPanel;
