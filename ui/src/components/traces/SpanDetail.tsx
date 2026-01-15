/**
 * Span Detail Component
 * Detailed view of a selected span
 */

import { X, Clock, Hash, Tag, AlertTriangle, FileText } from 'lucide-react';
import { format } from 'date-fns';
import type { Span, SpanEvent } from './SpanWaterfall';

interface SpanDetailProps {
  span: Span;
  onClose: () => void;
}

export default function SpanDetail({ span, onClose }: SpanDetailProps) {
  const formatTimestamp = (ms: number) => {
    return format(new Date(ms), 'HH:mm:ss.SSS');
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-800">
        <div className="min-w-0 flex-1">
          <h3 className="truncate font-medium text-gray-900 dark:text-white">
            {span.operationName}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {span.serviceName}
          </p>
        </div>
        <button
          onClick={onClose}
          className="ml-2 rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800 dark:hover:text-gray-300"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Status */}
        <div className="mb-6">
          <div
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 ${
              span.status === 'ok'
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
            }`}
          >
            {span.status === 'error' && <AlertTriangle className="h-4 w-4" />}
            <span className="font-medium capitalize">{span.status}</span>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="mb-6 grid grid-cols-2 gap-4">
          <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
              <Clock className="h-4 w-4" />
              <span className="text-xs font-medium uppercase">Duration</span>
            </div>
            <p className="mt-1 text-lg font-semibold text-gray-900 dark:text-white">
              {span.duration >= 1000
                ? `${(span.duration / 1000).toFixed(2)}s`
                : `${span.duration}ms`}
            </p>
          </div>

          <div className="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
              <Hash className="h-4 w-4" />
              <span className="text-xs font-medium uppercase">Span ID</span>
            </div>
            <p className="mt-1 truncate font-mono text-sm text-gray-900 dark:text-white">
              {span.spanId}
            </p>
          </div>
        </div>

        {/* Timestamps */}
        <div className="mb-6">
          <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
            <Clock className="h-4 w-4" />
            Timestamps
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Start Time</span>
              <span className="font-mono text-gray-900 dark:text-white">
                {formatTimestamp(span.startTime)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">End Time</span>
              <span className="font-mono text-gray-900 dark:text-white">
                {formatTimestamp(span.startTime + span.duration)}
              </span>
            </div>
          </div>
        </div>

        {/* Parent Span */}
        {span.parentSpanId && (
          <div className="mb-6">
            <h4 className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
              Parent Span
            </h4>
            <code className="rounded bg-gray-100 px-2 py-1 font-mono text-sm text-gray-700 dark:bg-gray-800 dark:text-gray-300">
              {span.parentSpanId}
            </code>
          </div>
        )}

        {/* Attributes */}
        {span.attributes && Object.keys(span.attributes).length > 0 && (
          <div className="mb-6">
            <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
              <Tag className="h-4 w-4" />
              Attributes
            </h4>
            <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
              <table className="w-full text-sm">
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {Object.entries(span.attributes).map(([key, value]) => (
                    <tr key={key}>
                      <td className="whitespace-nowrap bg-gray-50 px-3 py-2 font-medium text-gray-700 dark:bg-gray-800 dark:text-gray-300">
                        {key}
                      </td>
                      <td className="break-all px-3 py-2 text-gray-900 dark:text-white">
                        {value}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Events */}
        {span.events && span.events.length > 0 && (
          <div>
            <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
              <FileText className="h-4 w-4" />
              Events ({span.events.length})
            </h4>
            <div className="space-y-2">
              {span.events.map((event, index) => (
                <div
                  key={index}
                  className="rounded-lg border border-gray-200 p-3 dark:border-gray-700"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-900 dark:text-white">
                      {event.name}
                    </span>
                    <span className="font-mono text-xs text-gray-500 dark:text-gray-400">
                      {formatTimestamp(event.timestamp)}
                    </span>
                  </div>
                  {event.attributes && Object.keys(event.attributes).length > 0 && (
                    <div className="mt-2 space-y-1">
                      {Object.entries(event.attributes).map(([key, value]) => (
                        <div key={key} className="text-xs">
                          <span className="text-gray-500 dark:text-gray-400">{key}:</span>{' '}
                          <span className="text-gray-700 dark:text-gray-300">{value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
