/**
 * Drift Overview Component
 * Multi-section dashboard showing overall drift status
 */

import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import { clsx } from 'clsx';

export interface DriftScores {
  feature: number;
  embedding: number;
  concept: number;
  prediction: number;
}

interface DriftOverviewProps {
  modelId: string;
  scores: DriftScores;
  thresholds: DriftScores;
  lastUpdated: string;
  activeAlerts: number;
}

function getStatus(score: number, threshold: number): 'healthy' | 'warning' | 'critical' {
  if (score < threshold * 0.7) return 'healthy';
  if (score < threshold) return 'warning';
  return 'critical';
}

function StatusBadge({ status }: { status: 'healthy' | 'warning' | 'critical' }) {
  const styles = {
    healthy: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    warning: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    critical: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  };

  const icons = {
    healthy: CheckCircle,
    warning: AlertTriangle,
    critical: AlertTriangle,
  };

  const Icon = icons[status];

  return (
    <span className={clsx('inline-flex items-center gap-1 rounded-full px-3 py-1 text-sm font-medium', styles[status])}>
      <Icon className="h-4 w-4" />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function DriftScoreCard({
  label,
  score,
  threshold,
  trend,
}: {
  label: string;
  score: number;
  threshold: number;
  trend?: 'up' | 'down' | 'stable';
}) {
  const status = getStatus(score, threshold);
  const percentage = (score / threshold) * 100;

  const statusColors = {
    healthy: 'text-green-600 dark:text-green-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    critical: 'text-red-600 dark:text-red-400',
  };

  const barColors = {
    healthy: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500',
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
          {label}
        </span>
        {trend && (
          <span className={statusColors[status]}>
            {trend === 'up' && <TrendingUp className="h-4 w-4" />}
            {trend === 'down' && <TrendingDown className="h-4 w-4" />}
            {trend === 'stable' && <Activity className="h-4 w-4" />}
          </span>
        )}
      </div>
      <div className="mt-2 flex items-end gap-2">
        <span className={clsx('text-3xl font-bold', statusColors[status])}>
          {(score * 100).toFixed(1)}%
        </span>
        <span className="mb-1 text-sm text-gray-400">
          / {(threshold * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-3">
        <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
          <div
            className={clsx('h-full rounded-full transition-all', barColors[status])}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export default function DriftOverview({
  modelId,
  scores,
  thresholds,
  lastUpdated,
  activeAlerts,
}: DriftOverviewProps) {
  // Calculate overall status
  const overallStatus = (): 'healthy' | 'warning' | 'critical' => {
    const statuses = [
      getStatus(scores.feature, thresholds.feature),
      getStatus(scores.embedding, thresholds.embedding),
      getStatus(scores.concept, thresholds.concept),
      getStatus(scores.prediction, thresholds.prediction),
    ];

    if (statuses.includes('critical')) return 'critical';
    if (statuses.includes('warning')) return 'warning';
    return 'healthy';
  };

  return (
    <div className="space-y-6">
      {/* Overall Status Header */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-800 dark:bg-gray-900">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Model: {modelId}
            </h2>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Last updated: {new Date(lastUpdated).toLocaleString()}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge status={overallStatus()} />
            {activeAlerts > 0 && (
              <span className="flex items-center gap-1 rounded-full bg-red-100 px-3 py-1 text-sm font-medium text-red-700 dark:bg-red-900/30 dark:text-red-400">
                <AlertTriangle className="h-4 w-4" />
                {activeAlerts} active alert{activeAlerts > 1 ? 's' : ''}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Score Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <DriftScoreCard
          label="Feature Drift"
          score={scores.feature}
          threshold={thresholds.feature}
          trend="stable"
        />
        <DriftScoreCard
          label="Embedding Drift"
          score={scores.embedding}
          threshold={thresholds.embedding}
          trend="up"
        />
        <DriftScoreCard
          label="Concept Drift"
          score={scores.concept}
          threshold={thresholds.concept}
          trend="down"
        />
        <DriftScoreCard
          label="Prediction Drift"
          score={scores.prediction}
          threshold={thresholds.prediction}
          trend="stable"
        />
      </div>
    </div>
  );
}
