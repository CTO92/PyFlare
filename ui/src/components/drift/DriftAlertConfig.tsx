/**
 * Drift Alert Configuration Component
 * Configure alert thresholds and notification settings
 */

import { useState } from 'react';
import { Save, Bell, Sliders, AlertTriangle, TestTube } from 'lucide-react';

interface DriftAlertConfig {
  enabled: boolean;
  thresholds: {
    feature: number;
    embedding: number;
    concept: number;
    prediction: number;
  };
  evaluationWindow: string;
  notificationChannels: string[];
  minSeverity: 'low' | 'medium' | 'high' | 'critical';
}

interface DriftAlertConfigProps {
  modelId: string;
  currentConfig: DriftAlertConfig;
  onSave: (config: DriftAlertConfig) => Promise<void>;
  onTest?: () => Promise<void>;
}

const EVALUATION_WINDOWS = [
  { value: '15m', label: '15 minutes' },
  { value: '1h', label: '1 hour' },
  { value: '6h', label: '6 hours' },
  { value: '24h', label: '24 hours' },
  { value: '7d', label: '7 days' },
];

const NOTIFICATION_CHANNELS = [
  { id: 'email', label: 'Email', icon: '📧' },
  { id: 'slack', label: 'Slack', icon: '💬' },
  { id: 'pagerduty', label: 'PagerDuty', icon: '🔔' },
  { id: 'webhook', label: 'Webhook', icon: '🌐' },
];

export default function DriftAlertConfig({
  modelId,
  currentConfig,
  onSave,
  onTest,
}: DriftAlertConfigProps) {
  const [config, setConfig] = useState<DriftAlertConfig>(currentConfig);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleThresholdChange = (type: keyof DriftAlertConfig['thresholds'], value: number) => {
    setConfig((prev) => ({
      ...prev,
      thresholds: {
        ...prev.thresholds,
        [type]: value,
      },
    }));
  };

  const handleChannelToggle = (channelId: string) => {
    setConfig((prev) => ({
      ...prev,
      notificationChannels: prev.notificationChannels.includes(channelId)
        ? prev.notificationChannels.filter((c) => c !== channelId)
        : [...prev.notificationChannels, channelId],
    }));
  };

  const handleSave = async () => {
    setIsSaving(true);
    setSaveSuccess(false);
    try {
      await onSave(config);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to save config:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleTest = async () => {
    if (!onTest) return;
    setIsTesting(true);
    try {
      await onTest();
    } catch (error) {
      console.error('Test failed:', error);
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 p-4 dark:border-gray-800">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-pyflare-100 p-2 dark:bg-pyflare-900/30">
            <Sliders className="h-5 w-5 text-pyflare-600 dark:text-pyflare-400" />
          </div>
          <div>
            <h3 className="font-medium text-gray-900 dark:text-white">
              Alert Configuration
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Model: {modelId}
            </p>
          </div>
        </div>

        {/* Enable Toggle */}
        <label className="flex cursor-pointer items-center gap-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Alerts {config.enabled ? 'Enabled' : 'Disabled'}
          </span>
          <div className="relative">
            <input
              type="checkbox"
              checked={config.enabled}
              onChange={(e) => setConfig((prev) => ({ ...prev, enabled: e.target.checked }))}
              className="sr-only"
            />
            <div
              className={`h-6 w-11 rounded-full transition-colors ${
                config.enabled ? 'bg-pyflare-500' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            />
            <div
              className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                config.enabled ? 'translate-x-5' : ''
              }`}
            />
          </div>
        </label>
      </div>

      <div className="space-y-6 p-4">
        {/* Thresholds */}
        <div>
          <h4 className="mb-4 flex items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
            <AlertTriangle className="h-4 w-4" />
            Alert Thresholds
          </h4>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {Object.entries(config.thresholds).map(([type, value]) => (
              <div key={type}>
                <label className="mb-1 block text-sm text-gray-600 dark:text-gray-400">
                  {type.charAt(0).toUpperCase() + type.slice(1)} Drift
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={value * 100}
                    onChange={(e) =>
                      handleThresholdChange(
                        type as keyof DriftAlertConfig['thresholds'],
                        parseInt(e.target.value, 10) / 100
                      )
                    }
                    className="h-2 flex-1 cursor-pointer appearance-none rounded-lg bg-gray-200 accent-pyflare-500 dark:bg-gray-700"
                  />
                  <span className="w-16 rounded bg-gray-100 px-2 py-1 text-center text-sm font-medium dark:bg-gray-800">
                    {(value * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Evaluation Window */}
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-900 dark:text-white">
            Evaluation Window
          </label>
          <select
            value={config.evaluationWindow}
            onChange={(e) => setConfig((prev) => ({ ...prev, evaluationWindow: e.target.value }))}
            className="input w-full max-w-xs"
          >
            {EVALUATION_WINDOWS.map((window) => (
              <option key={window.value} value={window.value}>
                {window.label}
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Time window for calculating drift scores
          </p>
        </div>

        {/* Minimum Severity */}
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-900 dark:text-white">
            Minimum Alert Severity
          </label>
          <div className="flex flex-wrap gap-2">
            {(['low', 'medium', 'high', 'critical'] as const).map((severity) => (
              <button
                key={severity}
                onClick={() => setConfig((prev) => ({ ...prev, minSeverity: severity }))}
                className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                  config.minSeverity === severity
                    ? 'bg-pyflare-500 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'
                }`}
              >
                {severity.charAt(0).toUpperCase() + severity.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Notification Channels */}
        <div>
          <h4 className="mb-3 flex items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
            <Bell className="h-4 w-4" />
            Notification Channels
          </h4>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {NOTIFICATION_CHANNELS.map((channel) => (
              <label
                key={channel.id}
                className={`flex cursor-pointer items-center gap-2 rounded-lg border p-3 transition-colors ${
                  config.notificationChannels.includes(channel.id)
                    ? 'border-pyflare-500 bg-pyflare-50 dark:bg-pyflare-900/20'
                    : 'border-gray-200 hover:border-gray-300 dark:border-gray-700'
                }`}
              >
                <input
                  type="checkbox"
                  checked={config.notificationChannels.includes(channel.id)}
                  onChange={() => handleChannelToggle(channel.id)}
                  className="h-4 w-4 rounded border-gray-300 text-pyflare-600 focus:ring-pyflare-500"
                />
                <span className="text-lg">{channel.icon}</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {channel.label}
                </span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between border-t border-gray-200 p-4 dark:border-gray-800">
        <div>
          {saveSuccess && (
            <span className="text-sm text-green-600 dark:text-green-400">
              Configuration saved successfully!
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {onTest && (
            <button
              onClick={handleTest}
              disabled={isTesting}
              className="btn-secondary flex items-center gap-2"
            >
              <TestTube className="h-4 w-4" />
              {isTesting ? 'Testing...' : 'Test Alert'}
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="btn-primary flex items-center gap-2"
          >
            <Save className="h-4 w-4" />
            {isSaving ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </div>
    </div>
  );
}
