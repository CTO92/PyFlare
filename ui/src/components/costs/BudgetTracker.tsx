/**
 * Budget Tracker Component
 * Visual budget tracking with alerts and historical comparison
 */

import { AlertTriangle, TrendingUp, TrendingDown, Calendar, DollarSign } from 'lucide-react';
import { clsx } from 'clsx';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

export interface Budget {
  id: string;
  name: string;
  limit: number;
  used: number;
  period: 'daily' | 'weekly' | 'monthly';
  alertThreshold: number;
  resetDate: string;
  scope?: {
    type: 'model' | 'service' | 'user' | 'global';
    value?: string;
  };
}

export interface BudgetHistoryPoint {
  date: string;
  used: number;
  limit: number;
}

interface BudgetTrackerProps {
  budgets: Budget[];
  history?: BudgetHistoryPoint[];
  selectedBudgetId?: string;
  onBudgetSelect?: (budgetId: string) => void;
  currency?: string;
}

function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function BudgetCard({
  budget,
  currency,
  isSelected,
  onClick,
}: {
  budget: Budget;
  currency: string;
  isSelected: boolean;
  onClick: () => void;
}) {
  const percentage = (budget.used / budget.limit) * 100;
  const isOverBudget = percentage >= 100;
  const isNearLimit = percentage >= budget.alertThreshold;

  const periodLabels = {
    daily: 'Daily',
    weekly: 'Weekly',
    monthly: 'Monthly',
  };

  const scopeLabels = {
    model: 'Model',
    service: 'Service',
    user: 'User',
    global: 'Global',
  };

  return (
    <div
      className={clsx(
        'cursor-pointer rounded-lg border p-4 transition-all',
        isSelected
          ? 'border-pyflare-500 bg-pyflare-50 dark:bg-pyflare-900/20'
          : 'border-gray-200 bg-white hover:border-gray-300 dark:border-gray-800 dark:bg-gray-900 dark:hover:border-gray-700'
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">
            {budget.name}
          </h4>
          <div className="mt-1 flex items-center gap-2">
            <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-400">
              {periodLabels[budget.period]}
            </span>
            {budget.scope && (
              <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                {scopeLabels[budget.scope.type]}
                {budget.scope.value && `: ${budget.scope.value}`}
              </span>
            )}
          </div>
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
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold text-gray-900 dark:text-white">
            {formatCurrency(budget.used, currency)}
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            of {formatCurrency(budget.limit, currency)}
          </span>
        </div>

        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
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

        <div className="mt-2 flex items-center justify-between text-sm">
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
            {percentage.toFixed(1)}% used
          </span>
          <span className="text-gray-500 dark:text-gray-400">
            Resets: {format(new Date(budget.resetDate), 'MMM d')}
          </span>
        </div>
      </div>
    </div>
  );
}

function BudgetHistoryChart({
  history,
  budget,
  currency,
}: {
  history: BudgetHistoryPoint[];
  budget: Budget;
  currency: string;
}) {
  const formattedData = history.map((point) => ({
    ...point,
    date: format(new Date(point.date), 'MMM d'),
  }));

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{ value: number }>;
    label?: string;
  }) => {
    if (!active || !payload?.length) return null;

    return (
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg dark:border-gray-700 dark:bg-gray-800">
        <p className="text-sm font-medium text-gray-900 dark:text-white">{label}</p>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Used: {formatCurrency(payload[0].value, currency)}
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-500">
          Limit: {formatCurrency(budget.limit, currency)}
        </p>
      </div>
    );
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <h4 className="mb-4 font-medium text-gray-900 dark:text-white">
        Budget History: {budget.name}
      </h4>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={formattedData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
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
          <ReferenceLine
            y={budget.limit}
            stroke="#ef4444"
            strokeDasharray="5 5"
            label={{
              value: 'Limit',
              position: 'insideTopRight',
              fill: '#ef4444',
              fontSize: 12,
            }}
          />
          <ReferenceLine
            y={budget.limit * (budget.alertThreshold / 100)}
            stroke="#f59e0b"
            strokeDasharray="3 3"
            label={{
              value: 'Alert',
              position: 'insideTopRight',
              fill: '#f59e0b',
              fontSize: 10,
            }}
          />
          <Line
            type="monotone"
            dataKey="used"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: '#3b82f6', r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function BudgetTracker({
  budgets,
  history = [],
  selectedBudgetId,
  onBudgetSelect,
  currency = 'USD',
}: BudgetTrackerProps) {
  const selectedBudget = budgets.find((b) => b.id === selectedBudgetId) || budgets[0];

  // Summary stats
  const totalBudget = budgets.reduce((sum, b) => sum + b.limit, 0);
  const totalUsed = budgets.reduce((sum, b) => sum + b.used, 0);
  const budgetsAtRisk = budgets.filter(
    (b) => (b.used / b.limit) * 100 >= b.alertThreshold
  ).length;
  const budgetsExceeded = budgets.filter((b) => b.used >= b.limit).length;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
            <DollarSign className="h-4 w-4" />
            Total Budget
          </div>
          <p className="mt-1 text-xl font-bold text-gray-900 dark:text-white">
            {formatCurrency(totalBudget, currency)}
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
            <TrendingUp className="h-4 w-4" />
            Total Used
          </div>
          <p className="mt-1 text-xl font-bold text-gray-900 dark:text-white">
            {formatCurrency(totalUsed, currency)}
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4" />
            At Risk
          </div>
          <p className="mt-1 text-xl font-bold text-yellow-600 dark:text-yellow-400">
            {budgetsAtRisk}
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
            <TrendingDown className="h-4 w-4" />
            Exceeded
          </div>
          <p className="mt-1 text-xl font-bold text-red-600 dark:text-red-400">
            {budgetsExceeded}
          </p>
        </div>
      </div>

      {/* Budget Cards Grid */}
      <div>
        <h3 className="mb-4 font-medium text-gray-900 dark:text-white">
          Active Budgets
        </h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {budgets.map((budget) => (
            <BudgetCard
              key={budget.id}
              budget={budget}
              currency={currency}
              isSelected={budget.id === selectedBudget?.id}
              onClick={() => onBudgetSelect?.(budget.id)}
            />
          ))}
        </div>
      </div>

      {/* History Chart */}
      {selectedBudget && history.length > 0 && (
        <BudgetHistoryChart
          history={history}
          budget={selectedBudget}
          currency={currency}
        />
      )}

      {budgets.length === 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-8 text-center dark:border-gray-800 dark:bg-gray-900">
          <Calendar className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 font-medium text-gray-900 dark:text-white">
            No Budgets Configured
          </h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Set up cost budgets to track and control your spending.
          </p>
        </div>
      )}
    </div>
  );
}
