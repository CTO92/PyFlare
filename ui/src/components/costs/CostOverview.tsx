/**
 * Cost Overview Component
 * Summary dashboard showing key cost metrics and trends
 */

import { DollarSign, TrendingUp, TrendingDown, Zap, Clock, AlertTriangle } from 'lucide-react';
import { clsx } from 'clsx';

export interface CostMetrics {
  totalCost: number;
  costChange: number;
  totalTokens: number;
  tokenChange: number;
  avgCostPerRequest: number;
  avgCostChange: number;
  requestCount: number;
  requestChange: number;
}

export interface BudgetStatus {
  used: number;
  limit: number;
  period: 'daily' | 'weekly' | 'monthly';
  alertThreshold: number;
}

interface CostOverviewProps {
  metrics: CostMetrics;
  budget: BudgetStatus;
  timeRange: string;
  currency?: string;
}

function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatNumber(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString();
}

function MetricCard({
  label,
  value,
  change,
  icon: Icon,
  format = 'number',
  currency = 'USD',
}: {
  label: string;
  value: number;
  change: number;
  icon: React.ElementType;
  format?: 'currency' | 'number' | 'tokens';
  currency?: string;
}) {
  const isPositive = change > 0;
  const isNeutral = change === 0;

  const formattedValue =
    format === 'currency'
      ? formatCurrency(value, currency)
      : format === 'tokens'
        ? formatNumber(value)
        : formatNumber(value);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
          {label}
        </span>
        <Icon className="h-5 w-5 text-gray-400" />
      </div>
      <div className="mt-2 flex items-end justify-between">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">
          {formattedValue}
        </span>
        {!isNeutral && (
          <span
            className={clsx(
              'flex items-center text-sm font-medium',
              isPositive
                ? 'text-red-600 dark:text-red-400'
                : 'text-green-600 dark:text-green-400'
            )}
          >
            {isPositive ? (
              <TrendingUp className="mr-1 h-4 w-4" />
            ) : (
              <TrendingDown className="mr-1 h-4 w-4" />
            )}
            {Math.abs(change).toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

function BudgetProgress({ budget }: { budget: BudgetStatus }) {
  const percentage = (budget.used / budget.limit) * 100;
  const isOverBudget = percentage >= 100;
  const isNearLimit = percentage >= budget.alertThreshold;

  const periodLabels = {
    daily: 'Daily',
    weekly: 'Weekly',
    monthly: 'Monthly',
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white">
            {periodLabels[budget.period]} Budget
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {formatCurrency(budget.used)} of {formatCurrency(budget.limit)}
          </p>
        </div>
        {(isOverBudget || isNearLimit) && (
          <AlertTriangle
            className={clsx(
              'h-5 w-5',
              isOverBudget ? 'text-red-500' : 'text-yellow-500'
            )}
          />
        )}
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500 dark:text-gray-400">Usage</span>
          <span
            className={clsx(
              'font-medium',
              isOverBudget
                ? 'text-red-600 dark:text-red-400'
                : isNearLimit
                  ? 'text-yellow-600 dark:text-yellow-400'
                  : 'text-green-600 dark:text-green-400'
            )}
          >
            {percentage.toFixed(1)}%
          </span>
        </div>
        <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
          <div
            className={clsx(
              'h-full rounded-full transition-all',
              isOverBudget
                ? 'bg-red-500'
                : isNearLimit
                  ? 'bg-yellow-500'
                  : 'bg-green-500'
            )}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
        {budget.alertThreshold < 100 && (
          <div className="relative mt-1">
            <div
              className="absolute h-2 w-0.5 bg-gray-400"
              style={{ left: `${budget.alertThreshold}%` }}
            />
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between text-sm">
        <span className="text-gray-500 dark:text-gray-400">Remaining</span>
        <span
          className={clsx(
            'font-medium',
            isOverBudget
              ? 'text-red-600 dark:text-red-400'
              : 'text-gray-900 dark:text-white'
          )}
        >
          {isOverBudget ? '-' : ''}
          {formatCurrency(Math.abs(budget.limit - budget.used))}
        </span>
      </div>
    </div>
  );
}

export default function CostOverview({
  metrics,
  budget,
  timeRange,
  currency = 'USD',
}: CostOverviewProps) {
  return (
    <div className="space-y-6">
      {/* Time Range Indicator */}
      <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
        <Clock className="h-4 w-4" />
        Showing data for: {timeRange}
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Total Cost"
          value={metrics.totalCost}
          change={metrics.costChange}
          icon={DollarSign}
          format="currency"
          currency={currency}
        />
        <MetricCard
          label="Total Tokens"
          value={metrics.totalTokens}
          change={metrics.tokenChange}
          icon={Zap}
          format="tokens"
        />
        <MetricCard
          label="Avg Cost/Request"
          value={metrics.avgCostPerRequest}
          change={metrics.avgCostChange}
          icon={TrendingUp}
          format="currency"
          currency={currency}
        />
        <MetricCard
          label="Total Requests"
          value={metrics.requestCount}
          change={metrics.requestChange}
          icon={Clock}
          format="number"
        />
      </div>

      {/* Budget Progress */}
      <BudgetProgress budget={budget} />
    </div>
  );
}
