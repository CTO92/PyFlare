/**
 * Trace Comparison Component
 * Side-by-side comparison of two traces
 */

import { useMemo } from 'react';
import { ArrowRight, Clock, Hash, AlertTriangle, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import SpanWaterfall, { type Span } from './SpanWaterfall';

interface Trace {
  traceId: string;
  spans: Span[];
  totalDuration: number;
  service: string;
  model: string;
  status: 'ok' | 'error';
  timestamp: string;
  tokens?: { input: number; output: number };
  cost?: number;
}

interface TraceComparisonProps {
  leftTrace: Trace;
  rightTrace: Trace;
  diffMode?: 'structure' | 'timing' | 'attributes';
}

interface ComparisonMetric {
  label: string;
  leftValue: string | number;
  rightValue: string | number;
  diff?: number;
  diffPercent?: number;
  unit?: string;
}

function calculateDiff(left: number, right: number): { diff: number; percent: number } {
  const diff = right - left;
  const percent = left !== 0 ? ((right - left) / left) * 100 : 0;
  return { diff, percent };
}

function DiffIndicator({ diff, percent }: { diff: number; percent: number }) {
  if (Math.abs(percent) < 1) {
    return (
      <span className="flex items-center gap-1 text-gray-500">
        <Minus className="h-4 w-4" />
        <span className="text-sm">Same</span>
      </span>
    );
  }

  if (diff > 0) {
    return (
      <span className="flex items-center gap-1 text-red-600 dark:text-red-400">
        <TrendingUp className="h-4 w-4" />
        <span className="text-sm">+{percent.toFixed(1)}%</span>
      </span>
    );
  }

  return (
    <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
      <TrendingDown className="h-4 w-4" />
      <span className="text-sm">{percent.toFixed(1)}%</span>
    </span>
  );
}

export default function TraceComparison({
  leftTrace,
  rightTrace,
  diffMode = 'timing',
}: TraceComparisonProps) {
  const metrics: ComparisonMetric[] = useMemo(() => {
    const durationDiff = calculateDiff(leftTrace.totalDuration, rightTrace.totalDuration);
    const spanDiff = calculateDiff(leftTrace.spans.length, rightTrace.spans.length);

    const result: ComparisonMetric[] = [
      {
        label: 'Total Duration',
        leftValue: leftTrace.totalDuration,
        rightValue: rightTrace.totalDuration,
        diff: durationDiff.diff,
        diffPercent: durationDiff.percent,
        unit: 'ms',
      },
      {
        label: 'Span Count',
        leftValue: leftTrace.spans.length,
        rightValue: rightTrace.spans.length,
        diff: spanDiff.diff,
        diffPercent: spanDiff.percent,
      },
    ];

    if (leftTrace.tokens && rightTrace.tokens) {
      const inputDiff = calculateDiff(leftTrace.tokens.input, rightTrace.tokens.input);
      const outputDiff = calculateDiff(leftTrace.tokens.output, rightTrace.tokens.output);

      result.push(
        {
          label: 'Input Tokens',
          leftValue: leftTrace.tokens.input,
          rightValue: rightTrace.tokens.input,
          diff: inputDiff.diff,
          diffPercent: inputDiff.percent,
        },
        {
          label: 'Output Tokens',
          leftValue: leftTrace.tokens.output,
          rightValue: rightTrace.tokens.output,
          diff: outputDiff.diff,
          diffPercent: outputDiff.percent,
        }
      );
    }

    if (leftTrace.cost !== undefined && rightTrace.cost !== undefined) {
      const costDiff = calculateDiff(leftTrace.cost, rightTrace.cost);
      result.push({
        label: 'Cost',
        leftValue: `$${(leftTrace.cost / 1000000).toFixed(4)}`,
        rightValue: `$${(rightTrace.cost / 1000000).toFixed(4)}`,
        diff: costDiff.diff,
        diffPercent: costDiff.percent,
      });
    }

    return result;
  }, [leftTrace, rightTrace]);

  const formatValue = (value: string | number, unit?: string): string => {
    if (typeof value === 'string') return value;
    if (unit === 'ms') {
      return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value}ms`;
    }
    return String(value);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-center gap-4">
        <div className="flex-1 rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2">
            <Hash className="h-4 w-4 text-gray-400" />
            <code className="font-mono text-sm text-pyflare-600 dark:text-pyflare-400">
              {leftTrace.traceId.substring(0, 16)}...
            </code>
          </div>
          <div className="mt-2 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>{leftTrace.service}</span>
            <span>·</span>
            <span>{leftTrace.model}</span>
            <span>·</span>
            <span className={leftTrace.status === 'ok' ? 'text-green-600' : 'text-red-600'}>
              {leftTrace.status}
            </span>
          </div>
        </div>

        <ArrowRight className="h-6 w-6 flex-shrink-0 text-gray-400" />

        <div className="flex-1 rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
          <div className="flex items-center gap-2">
            <Hash className="h-4 w-4 text-gray-400" />
            <code className="font-mono text-sm text-pyflare-600 dark:text-pyflare-400">
              {rightTrace.traceId.substring(0, 16)}...
            </code>
          </div>
          <div className="mt-2 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
            <span>{rightTrace.service}</span>
            <span>·</span>
            <span>{rightTrace.model}</span>
            <span>·</span>
            <span className={rightTrace.status === 'ok' ? 'text-green-600' : 'text-red-600'}>
              {rightTrace.status}
            </span>
          </div>
        </div>
      </div>

      {/* Metrics Comparison */}
      <div className="rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
        <div className="border-b border-gray-200 px-4 py-3 dark:border-gray-800">
          <h3 className="font-medium text-gray-900 dark:text-white">
            Comparison Summary
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium uppercase text-gray-500">
                  Metric
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">
                  Left Trace
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">
                  Right Trace
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium uppercase text-gray-500">
                  Difference
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
              {metrics.map((metric) => (
                <tr key={metric.label}>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">
                    {metric.label}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-600 dark:text-gray-400">
                    {formatValue(metric.leftValue, metric.unit)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-600 dark:text-gray-400">
                    {formatValue(metric.rightValue, metric.unit)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {metric.diffPercent !== undefined && (
                      <DiffIndicator
                        diff={metric.diff!}
                        percent={metric.diffPercent}
                      />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Status Warning */}
      {(leftTrace.status !== rightTrace.status) && (
        <div className="flex items-center gap-2 rounded-lg bg-yellow-50 p-4 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-400">
          <AlertTriangle className="h-5 w-5" />
          <span>
            Status differs between traces: <strong>{leftTrace.status}</strong> vs <strong>{rightTrace.status}</strong>
          </span>
        </div>
      )}

      {/* Side-by-side Waterfalls */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div>
          <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
            <Clock className="h-4 w-4" />
            Left Trace Timeline
          </h4>
          <SpanWaterfall
            spans={leftTrace.spans}
            traceStart={Math.min(...leftTrace.spans.map((s) => s.startTime))}
            traceDuration={leftTrace.totalDuration}
          />
        </div>

        <div>
          <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
            <Clock className="h-4 w-4" />
            Right Trace Timeline
          </h4>
          <SpanWaterfall
            spans={rightTrace.spans}
            traceStart={Math.min(...rightTrace.spans.map((s) => s.startTime))}
            traceDuration={rightTrace.totalDuration}
          />
        </div>
      </div>
    </div>
  );
}
