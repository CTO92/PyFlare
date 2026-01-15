/**
 * Cost Forecast Component
 * Predictive cost analysis and projections
 */

import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts';
import { format, addDays, addWeeks, addMonths } from 'date-fns';
import { TrendingUp, TrendingDown, AlertTriangle, Target, Calendar } from 'lucide-react';
import { clsx } from 'clsx';

export interface ForecastDataPoint {
  date: string;
  actual?: number;
  forecast?: number;
  lowerBound?: number;
  upperBound?: number;
}

export interface ForecastMetrics {
  projectedEndOfPeriod: number;
  projectedChange: number;
  confidenceLevel: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  budgetImpact?: {
    budgetLimit: number;
    projectedUsage: number;
    willExceed: boolean;
    exceedDate?: string;
  };
}

interface CostForecastProps {
  historicalData: ForecastDataPoint[];
  forecastData: ForecastDataPoint[];
  metrics: ForecastMetrics;
  period: 'week' | 'month' | 'quarter';
  currency?: string;
  height?: number;
}

function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function MetricCard({
  label,
  value,
  subValue,
  icon: Icon,
  status,
}: {
  label: string;
  value: string;
  subValue?: string;
  icon: React.ElementType;
  status?: 'positive' | 'negative' | 'neutral' | 'warning';
}) {
  const statusColors = {
    positive: 'text-green-600 dark:text-green-400',
    negative: 'text-red-600 dark:text-red-400',
    neutral: 'text-gray-600 dark:text-gray-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
        <Icon className="h-4 w-4" />
        {label}
      </div>
      <p
        className={clsx(
          'mt-1 text-xl font-bold',
          status ? statusColors[status] : 'text-gray-900 dark:text-white'
        )}
      >
        {value}
      </p>
      {subValue && (
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {subValue}
        </p>
      )}
    </div>
  );
}

export default function CostForecast({
  historicalData,
  forecastData,
  metrics,
  period,
  currency = 'USD',
  height = 350,
}: CostForecastProps) {
  const [showConfidenceInterval, setShowConfidenceInterval] = useState(true);

  const combinedData = useMemo(() => {
    const historical = historicalData.map((d) => ({
      ...d,
      date: format(new Date(d.date), 'MMM d'),
      type: 'historical',
    }));

    const forecast = forecastData.map((d) => ({
      ...d,
      date: format(new Date(d.date), 'MMM d'),
      type: 'forecast',
    }));

    // Connect historical to forecast
    if (historical.length > 0 && forecast.length > 0) {
      forecast[0] = {
        ...forecast[0],
        actual: historical[historical.length - 1].actual,
      };
    }

    return [...historical, ...forecast];
  }, [historicalData, forecastData]);

  const periodLabels = {
    week: 'This Week',
    month: 'This Month',
    quarter: 'This Quarter',
  };

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ name: string; value: number; dataKey: string }>;
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;

    const dataPoint = payload[0]?.payload;
    const isHistorical = dataPoint?.type === 'historical';

    return (
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg dark:border-gray-700 dark:bg-gray-800">
        <p className="text-sm font-medium text-gray-900 dark:text-white">{label}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {isHistorical ? 'Historical' : 'Forecast'}
        </p>
        <div className="mt-2 space-y-1">
          {dataPoint?.actual !== undefined && (
            <p className="text-sm">
              <span className="text-gray-600 dark:text-gray-400">Actual: </span>
              <span className="font-medium text-blue-600 dark:text-blue-400">
                {formatCurrency(dataPoint.actual, currency)}
              </span>
            </p>
          )}
          {dataPoint?.forecast !== undefined && (
            <p className="text-sm">
              <span className="text-gray-600 dark:text-gray-400">Forecast: </span>
              <span className="font-medium text-purple-600 dark:text-purple-400">
                {formatCurrency(dataPoint.forecast, currency)}
              </span>
            </p>
          )}
          {showConfidenceInterval &&
            dataPoint?.lowerBound !== undefined &&
            dataPoint?.upperBound !== undefined && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Range: {formatCurrency(dataPoint.lowerBound, currency)} -{' '}
                {formatCurrency(dataPoint.upperBound, currency)}
              </p>
            )}
        </div>
      </div>
    );
  };

  const trendIcon =
    metrics.trend === 'increasing' ? (
      <TrendingUp className="h-4 w-4" />
    ) : metrics.trend === 'decreasing' ? (
      <TrendingDown className="h-4 w-4" />
    ) : null;

  const trendStatus =
    metrics.trend === 'increasing'
      ? 'negative'
      : metrics.trend === 'decreasing'
        ? 'positive'
        : 'neutral';

  return (
    <div className="space-y-6">
      {/* Metrics Summary */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          label={`Projected ${periodLabels[period]}`}
          value={formatCurrency(metrics.projectedEndOfPeriod, currency)}
          icon={Target}
        />
        <MetricCard
          label="Projected Change"
          value={`${metrics.projectedChange > 0 ? '+' : ''}${metrics.projectedChange.toFixed(1)}%`}
          subValue="vs. previous period"
          icon={metrics.trend === 'increasing' ? TrendingUp : TrendingDown}
          status={trendStatus}
        />
        <MetricCard
          label="Confidence Level"
          value={`${metrics.confidenceLevel}%`}
          icon={Calendar}
          status="neutral"
        />
        {metrics.budgetImpact && (
          <MetricCard
            label="Budget Status"
            value={
              metrics.budgetImpact.willExceed
                ? 'Will Exceed'
                : `${((metrics.budgetImpact.projectedUsage / metrics.budgetImpact.budgetLimit) * 100).toFixed(0)}% of limit`
            }
            subValue={
              metrics.budgetImpact.exceedDate
                ? `Expected: ${format(new Date(metrics.budgetImpact.exceedDate), 'MMM d')}`
                : undefined
            }
            icon={AlertTriangle}
            status={metrics.budgetImpact.willExceed ? 'warning' : 'positive'}
          />
        )}
      </div>

      {/* Forecast Chart */}
      <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-medium text-gray-900 dark:text-white">
            Cost Forecast
          </h3>
          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <input
              type="checkbox"
              checked={showConfidenceInterval}
              onChange={(e) => setShowConfidenceInterval(e.target.checked)}
              className="rounded border-gray-300 text-pyflare-600 focus:ring-pyflare-500"
            />
            Show confidence interval
          </label>
        </div>

        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart
            data={combinedData}
            margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12, fill: '#9ca3af' }}
              tickLine={{ stroke: '#9ca3af' }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#9ca3af' }}
              tickLine={{ stroke: '#9ca3af' }}
              tickFormatter={(value) => formatCurrency(value, currency)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => (
                <span className="text-sm text-gray-600 dark:text-gray-400">{value}</span>
              )}
            />

            {/* Budget limit reference line */}
            {metrics.budgetImpact && (
              <ReferenceLine
                y={metrics.budgetImpact.budgetLimit}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{
                  value: 'Budget Limit',
                  position: 'insideTopRight',
                  fill: '#ef4444',
                  fontSize: 12,
                }}
              />
            )}

            {/* Confidence interval */}
            {showConfidenceInterval && (
              <Area
                type="monotone"
                dataKey="upperBound"
                stroke="none"
                fill="#8b5cf6"
                fillOpacity={0.1}
                name="Upper Bound"
              />
            )}
            {showConfidenceInterval && (
              <Area
                type="monotone"
                dataKey="lowerBound"
                stroke="none"
                fill="#ffffff"
                fillOpacity={1}
                name="Lower Bound"
              />
            )}

            {/* Actual line */}
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual Cost"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6', r: 4 }}
              activeDot={{ r: 6 }}
            />

            {/* Forecast line */}
            <Line
              type="monotone"
              dataKey="forecast"
              name="Forecast"
              stroke="#8b5cf6"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#8b5cf6', r: 4 }}
              activeDot={{ r: 6 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Forecast Details */}
      <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
        <h4 className="font-medium text-gray-900 dark:text-white">
          Forecast Summary
        </h4>
        <div className="mt-3 space-y-2 text-sm text-gray-600 dark:text-gray-400">
          <p className="flex items-center gap-2">
            {trendIcon}
            Cost trend is{' '}
            <span
              className={clsx(
                'font-medium',
                metrics.trend === 'increasing'
                  ? 'text-red-600 dark:text-red-400'
                  : metrics.trend === 'decreasing'
                    ? 'text-green-600 dark:text-green-400'
                    : ''
              )}
            >
              {metrics.trend}
            </span>
          </p>
          <p>
            Projected end-of-period cost:{' '}
            <span className="font-medium text-gray-900 dark:text-white">
              {formatCurrency(metrics.projectedEndOfPeriod, currency)}
            </span>
          </p>
          <p>
            Forecast confidence:{' '}
            <span className="font-medium text-gray-900 dark:text-white">
              {metrics.confidenceLevel}%
            </span>
          </p>
          {metrics.budgetImpact?.willExceed && (
            <p className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
              <AlertTriangle className="h-4 w-4" />
              Budget of {formatCurrency(metrics.budgetImpact.budgetLimit, currency)} is
              projected to be exceeded
              {metrics.budgetImpact.exceedDate &&
                ` by ${format(new Date(metrics.budgetImpact.exceedDate), 'MMMM d')}`}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
