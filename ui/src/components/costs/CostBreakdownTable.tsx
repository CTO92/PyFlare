/**
 * Cost Breakdown Table Component
 * Detailed cost breakdown by model, service, or user
 */

import { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, ChevronRight, Download } from 'lucide-react';
import { clsx } from 'clsx';

export interface CostBreakdownItem {
  id: string;
  name: string;
  cost: number;
  tokens: number;
  requests: number;
  avgCostPerRequest: number;
  percentageOfTotal: number;
  children?: CostBreakdownItem[];
}

interface CostBreakdownTableProps {
  data: CostBreakdownItem[];
  groupBy: 'model' | 'service' | 'user' | 'feature';
  totalCost: number;
  currency?: string;
  onExport?: () => void;
}

type SortField = 'name' | 'cost' | 'tokens' | 'requests';
type SortDirection = 'asc' | 'desc';

function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(value);
}

function formatNumber(value: number): string {
  return value.toLocaleString();
}

function CostBar({ percentage }: { percentage: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-20 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          className="h-full rounded-full bg-pyflare-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 dark:text-gray-400">
        {percentage.toFixed(1)}%
      </span>
    </div>
  );
}

function ExpandableRow({
  item,
  currency,
  level = 0,
}: {
  item: CostBreakdownItem;
  currency: string;
  level?: number;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasChildren = item.children && item.children.length > 0;

  return (
    <>
      <tr
        className={clsx(
          'border-b border-gray-100 dark:border-gray-800',
          hasChildren && 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50'
        )}
        onClick={() => hasChildren && setIsExpanded(!isExpanded)}
      >
        <td className="px-4 py-3">
          <div
            className="flex items-center gap-2"
            style={{ paddingLeft: `${level * 24}px` }}
          >
            {hasChildren && (
              <span className="text-gray-400">
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </span>
            )}
            <span
              className={clsx(
                'font-medium',
                level === 0
                  ? 'text-gray-900 dark:text-white'
                  : 'text-gray-600 dark:text-gray-400'
              )}
            >
              {item.name}
            </span>
          </div>
        </td>
        <td className="px-4 py-3 text-right font-medium text-gray-900 dark:text-white">
          {formatCurrency(item.cost, currency)}
        </td>
        <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">
          {formatNumber(item.tokens)}
        </td>
        <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">
          {formatNumber(item.requests)}
        </td>
        <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">
          {formatCurrency(item.avgCostPerRequest, currency)}
        </td>
        <td className="px-4 py-3">
          <CostBar percentage={item.percentageOfTotal} />
        </td>
      </tr>
      {isExpanded &&
        item.children?.map((child) => (
          <ExpandableRow
            key={child.id}
            item={child}
            currency={currency}
            level={level + 1}
          />
        ))}
    </>
  );
}

export default function CostBreakdownTable({
  data,
  groupBy,
  totalCost,
  currency = 'USD',
  onExport,
}: CostBreakdownTableProps) {
  const [sortField, setSortField] = useState<SortField>('cost');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'cost':
          comparison = a.cost - b.cost;
          break;
        case 'tokens':
          comparison = a.tokens - b.tokens;
          break;
        case 'requests':
          comparison = a.requests - b.requests;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [data, sortField, sortDirection]);

  const groupLabels = {
    model: 'Model',
    service: 'Service',
    user: 'User',
    feature: 'Feature',
  };

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
      <div className="flex items-center justify-between border-b border-gray-200 p-4 dark:border-gray-800">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white">
            Cost Breakdown by {groupLabels[groupBy]}
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Total: {formatCurrency(totalCost, currency)} across {data.length}{' '}
            {groupLabels[groupBy].toLowerCase()}s
          </p>
        </div>
        {onExport && (
          <button
            onClick={onExport}
            className="btn-secondary flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
        )}
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
                  {groupLabels[groupBy]} <SortIcon field="name" />
                </span>
              </th>
              <th
                className="cursor-pointer px-4 py-3 text-right text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('cost')}
              >
                <span className="flex items-center justify-end gap-1">
                  Cost <SortIcon field="cost" />
                </span>
              </th>
              <th
                className="cursor-pointer px-4 py-3 text-right text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('tokens')}
              >
                <span className="flex items-center justify-end gap-1">
                  Tokens <SortIcon field="tokens" />
                </span>
              </th>
              <th
                className="cursor-pointer px-4 py-3 text-right text-xs font-medium uppercase text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                onClick={() => handleSort('requests')}
              >
                <span className="flex items-center justify-end gap-1">
                  Requests <SortIcon field="requests" />
                </span>
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium uppercase text-gray-500">
                Avg/Request
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium uppercase text-gray-500">
                % of Total
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((item) => (
              <ExpandableRow key={item.id} item={item} currency={currency} />
            ))}
          </tbody>
        </table>
      </div>

      {data.length === 0 && (
        <div className="py-8 text-center text-gray-500 dark:text-gray-400">
          No cost data available
        </div>
      )}
    </div>
  );
}
