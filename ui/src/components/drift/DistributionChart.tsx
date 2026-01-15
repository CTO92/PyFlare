/**
 * Distribution Chart Component
 * Comparison of reference and current distributions
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface DistributionChartProps {
  referenceData: number[];
  currentData: number[];
  featureName: string;
  featureType: 'numerical' | 'categorical' | 'embedding';
  labels?: string[];
}

export default function DistributionChart({
  referenceData,
  currentData,
  featureName,
  featureType,
  labels,
}: DistributionChartProps) {
  // Generate labels if not provided
  const generateLabels = () => {
    if (labels) return labels;

    if (featureType === 'categorical') {
      return referenceData.map((_, i) => `Category ${i + 1}`);
    }

    // For numerical features, generate bin labels
    return referenceData.map((_, i) => {
      const binStart = (i / referenceData.length) * 100;
      const binEnd = ((i + 1) / referenceData.length) * 100;
      return `${binStart.toFixed(0)}-${binEnd.toFixed(0)}%`;
    });
  };

  const chartData = referenceData.map((ref, i) => ({
    label: generateLabels()[i],
    reference: ref,
    current: currentData[i] ?? 0,
    diff: (currentData[i] ?? 0) - ref,
  }));

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ name: string; value: number; fill: string }>;
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
              style={{ backgroundColor: entry.fill }}
            />
            <span className="text-gray-600 dark:text-gray-400">
              {entry.name}:
            </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {(entry.value * 100).toFixed(2)}%
            </span>
          </div>
        ))}
        {payload.length === 2 && (
          <div className="mt-2 border-t border-gray-200 pt-2 text-sm dark:border-gray-700">
            <span className="text-gray-500">Difference: </span>
            <span
              className={
                payload[1].value - payload[0].value > 0
                  ? 'text-red-600 dark:text-red-400'
                  : 'text-green-600 dark:text-green-400'
              }
            >
              {payload[1].value - payload[0].value > 0 ? '+' : ''}
              {((payload[1].value - payload[0].value) * 100).toFixed(2)}%
            </span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">
            Distribution Comparison: {featureName}
          </h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Reference vs. Current distribution ({featureType})
          </p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded bg-blue-500" />
            <span className="text-gray-600 dark:text-gray-400">Reference</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded bg-orange-500" />
            <span className="text-gray-600 dark:text-gray-400">Current</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            tickLine={{ stroke: '#9ca3af' }}
            interval={0}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickLine={{ stroke: '#9ca3af' }}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar
            dataKey="reference"
            name="Reference"
            fill="#3b82f6"
            radius={[4, 4, 0, 0]}
          />
          <Bar
            dataKey="current"
            name="Current"
            fill="#f97316"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Statistics Summary */}
      <div className="mt-4 grid grid-cols-3 gap-4 rounded-lg bg-gray-100 p-3 dark:bg-gray-800">
        <div>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Max Deviation
          </span>
          <p className="mt-1 font-medium text-gray-900 dark:text-white">
            {(Math.max(...chartData.map((d) => Math.abs(d.diff))) * 100).toFixed(2)}%
          </p>
        </div>
        <div>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Mean Absolute Deviation
          </span>
          <p className="mt-1 font-medium text-gray-900 dark:text-white">
            {(
              (chartData.reduce((sum, d) => sum + Math.abs(d.diff), 0) /
                chartData.length) *
              100
            ).toFixed(2)}
            %
          </p>
        </div>
        <div>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Bins with Drift
          </span>
          <p className="mt-1 font-medium text-gray-900 dark:text-white">
            {chartData.filter((d) => Math.abs(d.diff) > 0.05).length} of{' '}
            {chartData.length}
          </p>
        </div>
      </div>
    </div>
  );
}
