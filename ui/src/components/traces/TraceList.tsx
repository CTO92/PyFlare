/**
 * Trace List Component
 * High-performance trace list with virtualization support
 */

import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  CheckCircle,
  XCircle,
  ChevronUp,
  ChevronDown,
  AlertTriangle,
  Zap,
  Clock,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { clsx } from 'clsx';

export interface TraceItem {
  id: string;
  traceId: string;
  service: string;
  operation: string;
  model: string;
  status: 'ok' | 'error';
  duration: number;
  tokens: { input: number; output: number };
  cost?: number;
  timestamp: string;
  hasDrift?: boolean;
  hasSafetyIssue?: boolean;
  evalScore?: number;
}

type SortField = 'timestamp' | 'duration' | 'cost' | 'tokens';
type SortDirection = 'asc' | 'desc';

interface TraceListProps {
  traces: TraceItem[];
  loading?: boolean;
  selectedTraceIds?: string[];
  onSelectionChange?: (ids: string[]) => void;
  onTraceClick?: (traceId: string) => void;
}

export default function TraceList({
  traces,
  loading = false,
  selectedTraceIds = [],
  onSelectionChange,
  onTraceClick,
}: TraceListProps) {
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedTraces = useMemo(() => {
    return [...traces].sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case 'timestamp':
          comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
          break;
        case 'duration':
          comparison = a.duration - b.duration;
          break;
        case 'cost':
          comparison = (a.cost ?? 0) - (b.cost ?? 0);
          break;
        case 'tokens':
          comparison = (a.tokens.input + a.tokens.output) - (b.tokens.input + b.tokens.output);
          break;
      }

      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [traces, sortField, sortDirection]);

  const toggleAll = () => {
    if (selectedTraceIds.length === traces.length) {
      onSelectionChange?.([]);
    } else {
      onSelectionChange?.(traces.map((t) => t.traceId));
    }
  };

  const toggleTrace = (traceId: string) => {
    if (selectedTraceIds.includes(traceId)) {
      onSelectionChange?.(selectedTraceIds.filter((id) => id !== traceId));
    } else {
      onSelectionChange?.([...selectedTraceIds, traceId]);
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? (
      <ChevronUp className="h-4 w-4" />
    ) : (
      <ChevronDown className="h-4 w-4" />
    );
  };

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatCost = (micros: number): string => {
    if (micros < 1000) return `$0.00`;
    return `$${(micros / 1000000).toFixed(4)}`;
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-pyflare-500 border-t-transparent" />
      </div>
    );
  }

  if (traces.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center text-gray-500 dark:text-gray-400">
        <Clock className="mb-4 h-12 w-12" />
        <p>No traces found</p>
        <p className="mt-1 text-sm">Try adjusting your filters or time range</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800">
      <table className="w-full">
        <thead className="border-b border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-gray-900">
          <tr>
            {onSelectionChange && (
              <th className="w-10 px-4 py-3">
                <input
                  type="checkbox"
                  checked={selectedTraceIds.length === traces.length && traces.length > 0}
                  onChange={toggleAll}
                  className="h-4 w-4 rounded border-gray-300 text-pyflare-600 focus:ring-pyflare-500"
                />
              </th>
            )}
            <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Trace ID
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Service / Operation
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Model
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
              Status
            </th>
            <th
              className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              onClick={() => handleSort('duration')}
            >
              <span className="flex items-center gap-1">
                Duration
                <SortIcon field="duration" />
              </span>
            </th>
            <th
              className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              onClick={() => handleSort('tokens')}
            >
              <span className="flex items-center gap-1">
                Tokens
                <SortIcon field="tokens" />
              </span>
            </th>
            <th
              className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              onClick={() => handleSort('cost')}
            >
              <span className="flex items-center gap-1">
                Cost
                <SortIcon field="cost" />
              </span>
            </th>
            <th
              className="cursor-pointer px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              onClick={() => handleSort('timestamp')}
            >
              <span className="flex items-center gap-1">
                Time
                <SortIcon field="timestamp" />
              </span>
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-800 dark:bg-gray-900">
          {sortedTraces.map((trace) => (
            <tr
              key={trace.traceId}
              className={clsx(
                'group transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/50',
                selectedTraceIds.includes(trace.traceId) && 'bg-pyflare-50 dark:bg-pyflare-900/10'
              )}
              onClick={() => onTraceClick?.(trace.traceId)}
            >
              {onSelectionChange && (
                <td className="w-10 px-4 py-3" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={selectedTraceIds.includes(trace.traceId)}
                    onChange={() => toggleTrace(trace.traceId)}
                    className="h-4 w-4 rounded border-gray-300 text-pyflare-600 focus:ring-pyflare-500"
                  />
                </td>
              )}
              <td className="whitespace-nowrap px-4 py-3">
                <Link
                  to={`/traces/${trace.traceId}`}
                  className="font-mono text-sm text-pyflare-600 hover:underline dark:text-pyflare-400"
                  onClick={(e) => e.stopPropagation()}
                >
                  {trace.traceId.substring(0, 12)}...
                </Link>
                <div className="mt-0.5 flex items-center gap-1">
                  {trace.hasDrift && (
                    <span title="Drift detected" className="text-yellow-500">
                      <AlertTriangle className="h-3 w-3" />
                    </span>
                  )}
                  {trace.hasSafetyIssue && (
                    <span title="Safety issue" className="text-red-500">
                      <AlertTriangle className="h-3 w-3" />
                    </span>
                  )}
                  {trace.evalScore !== undefined && trace.evalScore < 0.5 && (
                    <span title="Low evaluation score" className="text-orange-500">
                      <Zap className="h-3 w-3" />
                    </span>
                  )}
                </div>
              </td>
              <td className="px-4 py-3">
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  {trace.service}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {trace.operation}
                </div>
              </td>
              <td className="whitespace-nowrap px-4 py-3">
                <span className="rounded-full bg-gray-100 px-2 py-1 text-xs font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-300">
                  {trace.model}
                </span>
              </td>
              <td className="whitespace-nowrap px-4 py-3">
                {trace.status === 'ok' ? (
                  <span className="flex items-center gap-1 text-green-600">
                    <CheckCircle className="h-4 w-4" />
                    <span className="text-sm">OK</span>
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-red-600">
                    <XCircle className="h-4 w-4" />
                    <span className="text-sm">Error</span>
                  </span>
                )}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                {formatDuration(trace.duration)}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                <span title={`Input: ${trace.tokens.input}, Output: ${trace.tokens.output}`}>
                  {trace.tokens.input} / {trace.tokens.output}
                </span>
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                {trace.cost !== undefined ? formatCost(trace.cost) : '-'}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                <span title={new Date(trace.timestamp).toLocaleString()}>
                  {formatDistanceToNow(new Date(trace.timestamp), { addSuffix: true })}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
