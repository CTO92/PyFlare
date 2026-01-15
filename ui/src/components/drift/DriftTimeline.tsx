/**
 * Drift Timeline Component
 * Time series visualization of drift scores
 */

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

interface TimelinePoint {
  timestamp: string;
  featureDrift: number;
  embeddingDrift: number;
  conceptDrift: number;
  predictionDrift: number;
}

interface DriftTimelineProps {
  data: TimelinePoint[];
  threshold?: number;
  showThreshold?: boolean;
  height?: number;
  selectedTypes?: ('feature' | 'embedding' | 'concept' | 'prediction')[];
}

const DRIFT_COLORS = {
  feature: '#3b82f6', // blue
  embedding: '#8b5cf6', // purple
  concept: '#f59e0b', // amber
  prediction: '#10b981', // emerald
};

export default function DriftTimeline({
  data,
  threshold = 0.3,
  showThreshold = true,
  height = 300,
  selectedTypes = ['feature', 'embedding', 'concept', 'prediction'],
}: DriftTimelineProps) {
  const formattedData = useMemo(() => {
    return data.map((point) => ({
      ...point,
      timestamp: format(new Date(point.timestamp), 'MMM d, HH:mm'),
      featureDrift: point.featureDrift * 100,
      embeddingDrift: point.embeddingDrift * 100,
      conceptDrift: point.conceptDrift * 100,
      predictionDrift: point.predictionDrift * 100,
    }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ name: string; value: number; color: string }>;
    label?: string;
  }) => {
    if (!active || !payload) return null;

    return (
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg dark:border-gray-700 dark:bg-gray-800">
        <p className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
          {label}
        </p>
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-2 text-sm">
            <div
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-gray-600 dark:text-gray-400">{entry.name}:</span>
            <span className="font-medium text-gray-900 dark:text-white">
              {entry.value.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
        <p className="text-gray-500 dark:text-gray-400">No drift data available</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <h3 className="mb-4 font-medium text-gray-900 dark:text-white">
        Drift Score Timeline
      </h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={formattedData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
          <XAxis
            dataKey="timestamp"
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={{ stroke: '#9ca3af' }}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={{ stroke: '#9ca3af' }}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => (
              <span className="text-sm text-gray-600 dark:text-gray-400">{value}</span>
            )}
          />

          {showThreshold && (
            <ReferenceLine
              y={threshold * 100}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{
                value: 'Threshold',
                position: 'insideTopRight',
                fill: '#ef4444',
                fontSize: 12,
              }}
            />
          )}

          {selectedTypes.includes('feature') && (
            <Line
              type="monotone"
              dataKey="featureDrift"
              name="Feature Drift"
              stroke={DRIFT_COLORS.feature}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          )}
          {selectedTypes.includes('embedding') && (
            <Line
              type="monotone"
              dataKey="embeddingDrift"
              name="Embedding Drift"
              stroke={DRIFT_COLORS.embedding}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          )}
          {selectedTypes.includes('concept') && (
            <Line
              type="monotone"
              dataKey="conceptDrift"
              name="Concept Drift"
              stroke={DRIFT_COLORS.concept}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          )}
          {selectedTypes.includes('prediction') && (
            <Line
              type="monotone"
              dataKey="predictionDrift"
              name="Prediction Drift"
              stroke={DRIFT_COLORS.prediction}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
