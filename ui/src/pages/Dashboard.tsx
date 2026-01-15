import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  DollarSign,
  TrendingUp
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface StatsCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: React.ElementType;
}

function StatsCard({ title, value, change, changeType = 'neutral', icon: Icon }: StatsCardProps) {
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
            {title}
          </p>
          <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">
            {value}
          </p>
          {change && (
            <p className={`mt-1 text-sm ${
              changeType === 'positive' ? 'text-green-600' :
              changeType === 'negative' ? 'text-red-600' :
              'text-gray-500'
            }`}>
              {change}
            </p>
          )}
        </div>
        <div className="rounded-full bg-pyflare-100 p-3 dark:bg-pyflare-900/20">
          <Icon className="h-6 w-6 text-pyflare-600 dark:text-pyflare-400" />
        </div>
      </div>
    </div>
  );
}

// Mock data for demo
const mockChartData = [
  { time: '00:00', requests: 1200, latency: 45 },
  { time: '04:00', requests: 800, latency: 42 },
  { time: '08:00', requests: 2400, latency: 48 },
  { time: '12:00', requests: 3200, latency: 52 },
  { time: '16:00', requests: 2800, latency: 47 },
  { time: '20:00', requests: 1600, latency: 44 },
];

export default function Dashboard() {
  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Dashboard
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Overview of your AI/ML observability metrics
        </p>
      </div>

      {/* Stats Grid */}
      <div className="mb-8 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Requests"
          value="1.2M"
          change="+12.5% from last week"
          changeType="positive"
          icon={Activity}
        />
        <StatsCard
          title="Avg Latency"
          value="47ms"
          change="-3.2% from last week"
          changeType="positive"
          icon={TrendingUp}
        />
        <StatsCard
          title="Active Alerts"
          value="3"
          change="2 critical"
          changeType="negative"
          icon={AlertTriangle}
        />
        <StatsCard
          title="Total Cost"
          value="$1,234"
          change="+8.1% from last week"
          changeType="negative"
          icon={DollarSign}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Request Volume Chart */}
        <div className="card p-6">
          <h3 className="mb-4 text-lg font-medium text-gray-900 dark:text-white">
            Request Volume
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockChartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis
                  dataKey="time"
                  className="text-xs text-gray-500"
                  tick={{ fill: '#6b7280' }}
                />
                <YAxis
                  className="text-xs text-gray-500"
                  tick={{ fill: '#6b7280' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '0.5rem'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="requests"
                  stroke="#ed751c"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency Chart */}
        <div className="card p-6">
          <h3 className="mb-4 text-lg font-medium text-gray-900 dark:text-white">
            Average Latency
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockChartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis
                  dataKey="time"
                  className="text-xs text-gray-500"
                  tick={{ fill: '#6b7280' }}
                />
                <YAxis
                  className="text-xs text-gray-500"
                  tick={{ fill: '#6b7280' }}
                  unit="ms"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '0.5rem'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="latency"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="mt-6">
        <div className="card">
          <div className="border-b border-gray-200 px-6 py-4 dark:border-gray-800">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Recent Alerts
            </h3>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-800">
            <div className="flex items-center gap-4 px-6 py-4">
              <div className="h-2 w-2 rounded-full bg-red-500" />
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">
                  High drift detected on model-gpt4
                </p>
                <p className="text-sm text-gray-500">5 minutes ago</p>
              </div>
            </div>
            <div className="flex items-center gap-4 px-6 py-4">
              <div className="h-2 w-2 rounded-full bg-yellow-500" />
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">
                  Latency spike in embedding service
                </p>
                <p className="text-sm text-gray-500">23 minutes ago</p>
              </div>
            </div>
            <div className="flex items-center gap-4 px-6 py-4">
              <div className="h-2 w-2 rounded-full bg-red-500" />
              <div className="flex-1">
                <p className="font-medium text-gray-900 dark:text-white">
                  Budget threshold exceeded for user-123
                </p>
                <p className="text-sm text-gray-500">1 hour ago</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
