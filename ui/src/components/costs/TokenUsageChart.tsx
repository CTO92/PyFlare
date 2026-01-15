/**
 * Token Usage Chart Component
 * Visualizes token usage over time with input/output breakdown
 */

import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { format } from 'date-fns';

interface TokenUsageDataPoint {
  timestamp: string;
  inputTokens: number;
  outputTokens: number;
  cacheHitTokens?: number;
}

interface TokenUsageChartProps {
  data: TokenUsageDataPoint[];
  chartType?: 'area' | 'bar';
  height?: number;
  showCacheHits?: boolean;
}

const COLORS = {
  input: '#3b82f6', // blue
  output: '#10b981', // emerald
  cacheHit: '#8b5cf6', // purple
};

function formatNumber(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toString();
}

export default function TokenUsageChart({
  data,
  chartType = 'area',
  height = 300,
  showCacheHits = false,
}: TokenUsageChartProps) {
  const formattedData = useMemo(() => {
    return data.map((point) => ({
      ...point,
      timestamp: format(new Date(point.timestamp), 'MMM d, HH:mm'),
      total: point.inputTokens + point.outputTokens,
    }));
  }, [data]);

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ name: string; value: number; color: string }>;
    label?: string;
  }) => {
    if (!active || !payload) return null;

    const total = payload.reduce((sum, entry) => sum + entry.value, 0);

    return (
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg dark:border-gray-700 dark:bg-gray-800">
        <p className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
          {label}
        </p>
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center justify-between gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-600 dark:text-gray-400">{entry.name}:</span>
            </div>
            <span className="font-medium text-gray-900 dark:text-white">
              {formatNumber(entry.value)}
            </span>
          </div>
        ))}
        <div className="mt-2 border-t border-gray-200 pt-2 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-gray-600 dark:text-gray-400">Total:</span>
            <span className="font-bold text-gray-900 dark:text-white">
              {formatNumber(total)}
            </span>
          </div>
        </div>
      </div>
    );
  };

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
        <p className="text-gray-500 dark:text-gray-400">No token usage data available</p>
      </div>
    );
  }

  const ChartComponent = chartType === 'area' ? AreaChart : BarChart;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <h3 className="mb-4 font-medium text-gray-900 dark:text-white">
        Token Usage Over Time
      </h3>
      <ResponsiveContainer width="100%" height={height}>
        {chartType === 'area' ? (
          <AreaChart data={formattedData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
            <XAxis
              dataKey="timestamp"
              tick={{ fontSize: 12, fill: '#9ca3af' }}
              tickLine={{ stroke: '#9ca3af' }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#9ca3af' }}
              tickLine={{ stroke: '#9ca3af' }}
              tickFormatter={formatNumber}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => (
                <span className="text-sm text-gray-600 dark:text-gray-400">{value}</span>
              )}
            />
            <Area
              type="monotone"
              dataKey="inputTokens"
              name="Input Tokens"
              stackId="1"
              stroke={COLORS.input}
              fill={COLORS.input}
              fillOpacity={0.6}
            />
            <Area
              type="monotone"
              dataKey="outputTokens"
              name="Output Tokens"
              stackId="1"
              stroke={COLORS.output}
              fill={COLORS.output}
              fillOpacity={0.6}
            />
            {showCacheHits && (
              <Area
                type="monotone"
                dataKey="cacheHitTokens"
                name="Cache Hits"
                stackId="2"
                stroke={COLORS.cacheHit}
                fill={COLORS.cacheHit}
                fillOpacity={0.4}
              />
            )}
          </AreaChart>
        ) : (
          <BarChart data={formattedData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
            <XAxis
              dataKey="timestamp"
              tick={{ fontSize: 12, fill: '#9ca3af' }}
              tickLine={{ stroke: '#9ca3af' }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#9ca3af' }}
              tickLine={{ stroke: '#9ca3af' }}
              tickFormatter={formatNumber}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => (
                <span className="text-sm text-gray-600 dark:text-gray-400">{value}</span>
              )}
            />
            <Bar
              dataKey="inputTokens"
              name="Input Tokens"
              stackId="1"
              fill={COLORS.input}
            />
            <Bar
              dataKey="outputTokens"
              name="Output Tokens"
              stackId="1"
              fill={COLORS.output}
            />
            {showCacheHits && (
              <Bar
                dataKey="cacheHitTokens"
                name="Cache Hits"
                stackId="2"
                fill={COLORS.cacheHit}
              />
            )}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
