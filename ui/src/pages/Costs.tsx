/**
 * Costs Page
 * Comprehensive cost analytics and budget management
 */

import { useState, useMemo } from 'react';
import { Settings, Download, RefreshCw, Calendar, BarChart3, TrendingUp } from 'lucide-react';
import CostOverview, { type CostMetrics, type BudgetStatus } from '../components/costs/CostOverview';
import CostBreakdownTable, { type CostBreakdownItem } from '../components/costs/CostBreakdownTable';
import TokenUsageChart from '../components/costs/TokenUsageChart';
import BudgetTracker, { type Budget, type BudgetHistoryPoint } from '../components/costs/BudgetTracker';
import BudgetConfig from '../components/costs/BudgetConfig';
import CostForecast, { type ForecastDataPoint, type ForecastMetrics } from '../components/costs/CostForecast';
import { useCosts } from '../hooks/useCosts';

type TabType = 'overview' | 'breakdown' | 'budgets' | 'forecast' | 'settings';
type TimeRange = '24h' | '7d' | '30d' | '90d';
type GroupBy = 'model' | 'service' | 'user' | 'feature';

// Mock data - in production this would come from API
const mockMetrics: CostMetrics = {
  totalCost: 12450.75,
  costChange: 12.5,
  totalTokens: 45_230_000,
  tokenChange: 8.2,
  avgCostPerRequest: 0.0245,
  avgCostChange: -3.1,
  requestCount: 508_600,
  requestChange: 15.7,
};

const mockBudgetStatus: BudgetStatus = {
  used: 9800,
  limit: 15000,
  period: 'monthly',
  alertThreshold: 80,
};

const mockBreakdownData: CostBreakdownItem[] = [
  {
    id: '1',
    name: 'gpt-4',
    cost: 5234.50,
    tokens: 15_200_000,
    requests: 156_000,
    avgCostPerRequest: 0.0335,
    percentageOfTotal: 42.0,
    children: [
      { id: '1a', name: 'chat-service', cost: 3500.00, tokens: 10_200_000, requests: 102_000, avgCostPerRequest: 0.0343, percentageOfTotal: 28.1 },
      { id: '1b', name: 'analysis-service', cost: 1734.50, tokens: 5_000_000, requests: 54_000, avgCostPerRequest: 0.0321, percentageOfTotal: 13.9 },
    ],
  },
  {
    id: '2',
    name: 'gpt-3.5-turbo',
    cost: 2890.25,
    tokens: 18_500_000,
    requests: 245_000,
    avgCostPerRequest: 0.0118,
    percentageOfTotal: 23.2,
  },
  {
    id: '3',
    name: 'claude-3-opus',
    cost: 2456.00,
    tokens: 6_800_000,
    requests: 68_000,
    avgCostPerRequest: 0.0361,
    percentageOfTotal: 19.7,
  },
  {
    id: '4',
    name: 'claude-3-sonnet',
    cost: 1870.00,
    tokens: 4_730_000,
    requests: 39_600,
    avgCostPerRequest: 0.0472,
    percentageOfTotal: 15.0,
  },
];

const mockTokenUsageData = Array.from({ length: 24 }, (_, i) => ({
  timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
  inputTokens: 500000 + Math.random() * 300000,
  outputTokens: 200000 + Math.random() * 150000,
  cacheHitTokens: 50000 + Math.random() * 30000,
}));

const mockBudgets: Budget[] = [
  {
    id: 'budget-1',
    name: 'Production API',
    limit: 15000,
    used: 9800,
    period: 'monthly',
    alertThreshold: 80,
    resetDate: new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1).toISOString(),
    scope: { type: 'global' },
  },
  {
    id: 'budget-2',
    name: 'GPT-4 Budget',
    limit: 6000,
    used: 5234.50,
    period: 'monthly',
    alertThreshold: 85,
    resetDate: new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1).toISOString(),
    scope: { type: 'model', value: 'gpt-4' },
  },
  {
    id: 'budget-3',
    name: 'Daily Cap',
    limit: 500,
    used: 412,
    period: 'daily',
    alertThreshold: 90,
    resetDate: new Date(new Date().setHours(24, 0, 0, 0)).toISOString(),
    scope: { type: 'global' },
  },
];

const mockBudgetHistory: BudgetHistoryPoint[] = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
  used: 5000 + Math.random() * 8000,
  limit: 15000,
}));

const mockHistoricalData: ForecastDataPoint[] = Array.from({ length: 14 }, (_, i) => ({
  date: new Date(Date.now() - (13 - i) * 24 * 60 * 60 * 1000).toISOString(),
  actual: 400 + Math.random() * 200 + i * 10,
}));

const mockForecastData: ForecastDataPoint[] = Array.from({ length: 7 }, (_, i) => {
  const base = 600 + i * 15;
  return {
    date: new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000).toISOString(),
    forecast: base,
    lowerBound: base * 0.85,
    upperBound: base * 1.15,
  };
});

const mockForecastMetrics: ForecastMetrics = {
  projectedEndOfPeriod: 14200,
  projectedChange: 12.5,
  confidenceLevel: 85,
  trend: 'increasing',
  budgetImpact: {
    budgetLimit: 15000,
    projectedUsage: 14200,
    willExceed: false,
  },
};

const availableModels = ['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet'];
const availableServices = ['chat-service', 'analysis-service', 'embedding-service'];

export default function Costs() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [groupBy, setGroupBy] = useState<GroupBy>('model');
  const [selectedBudgetId, setSelectedBudgetId] = useState<string | undefined>(mockBudgets[0]?.id);

  const { costs, loading, error, refetch } = useCosts({ timeRange });

  const timeRangeLabel = {
    '24h': 'Last 24 Hours',
    '7d': 'Last 7 Days',
    '30d': 'Last 30 Days',
    '90d': 'Last 90 Days',
  };

  const handleExportBreakdown = () => {
    const data = mockBreakdownData.map((item) => ({
      name: item.name,
      cost: item.cost,
      tokens: item.tokens,
      requests: item.requests,
      avgCostPerRequest: item.avgCostPerRequest,
      percentageOfTotal: item.percentageOfTotal,
    }));

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cost-breakdown-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSaveBudgets = async (budgets: Budget[]) => {
    console.log('Saving budgets:', budgets);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  };

  const handleDeleteBudget = async (budgetId: string) => {
    console.log('Deleting budget:', budgetId);
    await new Promise((resolve) => setTimeout(resolve, 500));
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Cost Analytics
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Track, analyze, and optimize your AI/ML infrastructure costs
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Time Range Selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as TimeRange)}
            className="input"
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
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
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'breakdown', label: 'Cost Breakdown', icon: BarChart3 },
            { id: 'budgets', label: 'Budgets', icon: Calendar },
            { id: 'forecast', label: 'Forecast', icon: TrendingUp },
            { id: 'settings', label: 'Budget Settings', icon: Settings },
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
              <tab.icon className="h-4 w-4" />
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
          {/* Cost Overview */}
          <CostOverview
            metrics={costs?.metrics ?? mockMetrics}
            budget={mockBudgetStatus}
            timeRange={timeRangeLabel[timeRange]}
          />

          {/* Token Usage Chart */}
          <TokenUsageChart
            data={mockTokenUsageData}
            chartType="area"
            height={300}
            showCacheHits
          />
        </div>
      )}

      {activeTab === 'breakdown' && (
        <div className="space-y-6">
          {/* Group By Selector */}
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500 dark:text-gray-400">Group by:</span>
            <div className="flex rounded-lg border border-gray-200 dark:border-gray-700">
              {(['model', 'service', 'user', 'feature'] as const).map((g) => (
                <button
                  key={g}
                  onClick={() => setGroupBy(g)}
                  className={`px-4 py-2 text-sm ${
                    groupBy === g
                      ? 'bg-pyflare-50 text-pyflare-600 dark:bg-pyflare-900/20 dark:text-pyflare-400'
                      : 'text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800'
                  }`}
                >
                  {g.charAt(0).toUpperCase() + g.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Breakdown Table */}
          <CostBreakdownTable
            data={mockBreakdownData}
            groupBy={groupBy}
            totalCost={mockMetrics.totalCost}
            onExport={handleExportBreakdown}
          />

          {/* Token Usage Bar Chart */}
          <TokenUsageChart
            data={mockTokenUsageData}
            chartType="bar"
            height={250}
          />
        </div>
      )}

      {activeTab === 'budgets' && (
        <BudgetTracker
          budgets={mockBudgets}
          history={mockBudgetHistory}
          selectedBudgetId={selectedBudgetId}
          onBudgetSelect={setSelectedBudgetId}
        />
      )}

      {activeTab === 'forecast' && (
        <CostForecast
          historicalData={mockHistoricalData}
          forecastData={mockForecastData}
          metrics={mockForecastMetrics}
          period="month"
        />
      )}

      {activeTab === 'settings' && (
        <BudgetConfig
          budgets={mockBudgets}
          onSave={handleSaveBudgets}
          onDelete={handleDeleteBudget}
          availableModels={availableModels}
          availableServices={availableServices}
        />
      )}
    </div>
  );
}
