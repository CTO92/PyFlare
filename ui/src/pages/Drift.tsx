/**
 * Drift Page
 * Comprehensive drift detection dashboard
 */

import { useState } from 'react';
import { Settings, RefreshCw } from 'lucide-react';
import DriftOverview, { type DriftScores } from '../components/drift/DriftOverview';
import DriftTimeline from '../components/drift/DriftTimeline';
import FeatureDriftBreakdown, { type FeatureDrift } from '../components/drift/FeatureDriftBreakdown';
import DriftAlertConfig from '../components/drift/DriftAlertConfig';
import { useDrift } from '../hooks/useDrift';

// Mock data - in production this would come from API
const mockDriftScores: DriftScores = {
  feature: 0.25,
  embedding: 0.35,
  concept: 0.15,
  prediction: 0.2,
};

const mockThresholds: DriftScores = {
  feature: 0.3,
  embedding: 0.3,
  concept: 0.4,
  prediction: 0.3,
};

const mockTimelineData = Array.from({ length: 24 }, (_, i) => ({
  timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
  featureDrift: 0.2 + Math.random() * 0.15,
  embeddingDrift: 0.25 + Math.random() * 0.2,
  conceptDrift: 0.1 + Math.random() * 0.1,
  predictionDrift: 0.15 + Math.random() * 0.12,
}));

const mockFeatures: FeatureDrift[] = [
  {
    name: 'user_age',
    type: 'numerical',
    driftScore: 0.42,
    pValue: 0.001,
    referenceDistribution: [0.05, 0.1, 0.2, 0.25, 0.2, 0.12, 0.08],
    currentDistribution: [0.08, 0.15, 0.18, 0.22, 0.18, 0.12, 0.07],
    trend: 'increasing',
    importance: 0.85,
  },
  {
    name: 'session_duration',
    type: 'numerical',
    driftScore: 0.28,
    pValue: 0.023,
    referenceDistribution: [0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05],
    currentDistribution: [0.12, 0.18, 0.22, 0.18, 0.15, 0.1, 0.05],
    trend: 'stable',
    importance: 0.72,
  },
  {
    name: 'device_type',
    type: 'categorical',
    driftScore: 0.18,
    pValue: 0.089,
    referenceDistribution: [0.45, 0.35, 0.15, 0.05],
    currentDistribution: [0.42, 0.38, 0.14, 0.06],
    trend: 'stable',
    importance: 0.45,
  },
  {
    name: 'input_embedding',
    type: 'embedding',
    driftScore: 0.35,
    pValue: 0.003,
    referenceDistribution: [0.08, 0.12, 0.18, 0.22, 0.2, 0.12, 0.08],
    currentDistribution: [0.1, 0.14, 0.2, 0.18, 0.18, 0.12, 0.08],
    trend: 'increasing',
    importance: 0.92,
  },
  {
    name: 'request_count',
    type: 'numerical',
    driftScore: 0.12,
    pValue: 0.145,
    referenceDistribution: [0.2, 0.25, 0.22, 0.18, 0.1, 0.05],
    currentDistribution: [0.22, 0.24, 0.21, 0.17, 0.11, 0.05],
    trend: 'decreasing',
    importance: 0.38,
  },
];

const mockAlertConfig = {
  enabled: true,
  thresholds: mockThresholds,
  evaluationWindow: '1h',
  notificationChannels: ['slack', 'email'],
  minSeverity: 'medium' as const,
};

type TabType = 'overview' | 'features' | 'settings';

export default function Drift() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [selectedModel, setSelectedModel] = useState('gpt-4');

  const { driftStatus, loading, error, refetch } = useDrift({ modelId: selectedModel });

  const models = ['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet'];

  const handleSaveConfig = async (config: typeof mockAlertConfig) => {
    // In production, save to API
    console.log('Saving config:', config);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  };

  const handleTestAlert = async () => {
    // In production, trigger test alert
    console.log('Testing alert');
    await new Promise((resolve) => setTimeout(resolve, 1500));
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Drift Detection
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Monitor and analyze model drift across your ML systems
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Model Selector */}
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="input"
          >
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>

          {/* Refresh Button */}
          <button
            onClick={() => refetch()}
            disabled={loading}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6 border-b border-gray-200 dark:border-gray-800">
        <nav className="flex gap-6">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'features', label: 'Feature Analysis' },
            { id: 'settings', label: 'Alert Settings', icon: Settings },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`flex items-center gap-2 border-b-2 pb-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border-pyflare-500 text-pyflare-600 dark:text-pyflare-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              {tab.icon && <tab.icon className="h-4 w-4" />}
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-6 rounded-lg bg-red-50 p-4 text-red-700 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Overview Cards */}
          <DriftOverview
            modelId={selectedModel}
            scores={driftStatus?.driftScores ? {
              feature: driftStatus.driftScores.feature?.score ?? mockDriftScores.feature,
              embedding: driftStatus.driftScores.embedding?.score ?? mockDriftScores.embedding,
              concept: driftStatus.driftScores.concept?.score ?? mockDriftScores.concept,
              prediction: driftStatus.driftScores.prediction?.score ?? mockDriftScores.prediction,
            } : mockDriftScores}
            thresholds={mockThresholds}
            lastUpdated={driftStatus?.lastUpdated ?? new Date().toISOString()}
            activeAlerts={2}
          />

          {/* Timeline Chart */}
          <DriftTimeline
            data={mockTimelineData}
            threshold={0.3}
            showThreshold
            height={350}
          />
        </div>
      )}

      {activeTab === 'features' && (
        <FeatureDriftBreakdown
          features={mockFeatures}
          onFeatureSelect={(name) => console.log('Selected feature:', name)}
        />
      )}

      {activeTab === 'settings' && (
        <DriftAlertConfig
          modelId={selectedModel}
          currentConfig={mockAlertConfig}
          onSave={handleSaveConfig}
          onTest={handleTestAlert}
        />
      )}
    </div>
  );
}
