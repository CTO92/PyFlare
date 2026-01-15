/**
 * Feature Drift Breakdown Component
 * Individual feature-level drift analysis
 */

import { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, ArrowUpRight, ArrowDownRight, Minus, BarChart3 } from 'lucide-react';
import { clsx } from 'clsx';
import DistributionChart from './DistributionChart';

export interface FeatureDrift {
  name: string;
  type: 'numerical' | 'categorical' | 'embedding';
  driftScore: number;
  pValue: number;
  referenceDistribution: number[];
  currentDistribution: number[];
  trend: 'stable' | 'increasing' | 'decreasing';
  importance: number;
}

interface FeatureDriftBreakdownProps {
  features: FeatureDrift[];
  onFeatureSelect?: (featureName: string) => void;
}

type SortField = 'name' | 'driftScore' | 'importance' | 'pValue';
type SortDirection = 'asc' | 'desc';

function TrendIcon({ trend }: { trend: 'stable' | 'increasing' | 'decreasing' }) {
  switch (trend) {
    case 'increasing':
      return <ArrowUpRight className="h-4 w-4 text-red-500" />;
    case 'decreasing':
      return <ArrowDownRight className="h-4 w-4 text-green-500" />;
    default:
      return <Minus className="h-4 w-4 text-gray-400" />;
  }
}

function DriftScoreBadge({ score, threshold = 0.3 }: { score: number; threshold?: number }) {
  const status = score > threshold ? 'critical' : score > threshold * 0.7 ? 'warning' : 'healthy';

  const styles = {
    healthy: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    warning: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    critical: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  };

  return (
    <span className={clsx('rounded-full px-2 py-0.5 text-sm font-medium', styles[status])}>
      {(score * 100).toFixed(1)}%
    </span>
  );
}

export default function FeatureDriftBreakdown({
  features,
  onFeatureSelect,
}: FeatureDriftBreakdownProps) {
  const [sortField, setSortField] = useState<SortField>('driftScore');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [expandedFeature, setExpandedFeature] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'drifted' | 'stable'>('all');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const filteredAndSortedFeatures = useMemo(() => {
    let filtered = features;

    if (filter === 'drifted') {
      filtered = features.filter((f) => f.driftScore > 0.3);
    } else if (filter === 'stable') {
      filtered = features.filter((f) => f.driftScore <= 0.3);
    }

    return [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'driftScore':
          comparison = a.driftScore - b.driftScore;
          break;
        case 'importance':
          comparison = a.importance - b.importance;
          break;
        case 'pValue':
          comparison = a.pValue - b.pValue;
          break;
      }

      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [features, sortField, sortDirection, filter]);

  const driftedCount = features.filter((f) => f.driftScore > 0.3).length;

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? (
      <ChevronUp className="h-4 w-4" />
    ) : (
      <ChevronDown className="h-4 w-4" />
    );
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-gray-200 p-4 dark:border-gray-800">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white">
            Feature Drift Breakdown
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {features.length} features, {driftedCount} with significant drift
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex rounded-lg border border-gray-200 dark:border-gray-700">
          {(['all', 'drifted', 'stable'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={clsx(
                'px-4 py-2 text-sm',
                filter === f
                  ? 'bg-pyflare-50 text-pyflare-600 dark:bg-pyflare-900/20 dark:text-pyflare-400'
                  : 'text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800'
              )}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
              {f === 'drifted' && driftedCount > 0 && (
                <span className="ml-1 rounded-full bg-red-500 px-1.5 text-xs text-white">
                  {driftedCount}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th
                className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('name')}
              >
                <span className="flex items-center gap-1">
                  Feature <SortIcon field="name" />
                </span>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Type
              </th>
              <th
                className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('driftScore')}
              >
                <span className="flex items-center gap-1">
                  Drift Score <SortIcon field="driftScore" />
                </span>
              </th>
              <th
                className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('pValue')}
              >
                <span className="flex items-center gap-1">
                  P-Value <SortIcon field="pValue" />
                </span>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Trend
              </th>
              <th
                className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('importance')}
              >
                <span className="flex items-center gap-1">
                  Importance <SortIcon field="importance" />
                </span>
              </th>
              <th className="w-10 px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
            {filteredAndSortedFeatures.map((feature) => (
              <>
                <tr
                  key={feature.name}
                  className={clsx(
                    'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50',
                    expandedFeature === feature.name && 'bg-gray-50 dark:bg-gray-800/50'
                  )}
                  onClick={() => {
                    setExpandedFeature(expandedFeature === feature.name ? null : feature.name);
                    onFeatureSelect?.(feature.name);
                  }}
                >
                  <td className="px-4 py-3">
                    <span className="font-medium text-gray-900 dark:text-white">
                      {feature.name}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                      {feature.type}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <DriftScoreBadge score={feature.driftScore} />
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {feature.pValue < 0.001 ? '< 0.001' : feature.pValue.toFixed(3)}
                  </td>
                  <td className="px-4 py-3">
                    <TrendIcon trend={feature.trend} />
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-16 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                        <div
                          className="h-full rounded-full bg-pyflare-500"
                          style={{ width: `${feature.importance * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {(feature.importance * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <BarChart3 className="h-4 w-4 text-gray-400" />
                  </td>
                </tr>
                {/* Expanded Distribution View */}
                {expandedFeature === feature.name && (
                  <tr>
                    <td colSpan={7} className="bg-gray-50 px-4 py-4 dark:bg-gray-800/50">
                      <DistributionChart
                        referenceData={feature.referenceDistribution}
                        currentData={feature.currentDistribution}
                        featureName={feature.name}
                        featureType={feature.type}
                      />
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {filteredAndSortedFeatures.length === 0 && (
        <div className="py-8 text-center text-gray-500 dark:text-gray-400">
          No features match the current filter
        </div>
      )}
    </div>
  );
}
