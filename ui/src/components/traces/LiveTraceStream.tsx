/**
 * Live Trace Stream Component
 * Real-time trace streaming with WebSocket
 */

import { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Play, Pause, Trash2, CheckCircle, XCircle, Wifi, WifiOff } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useTraceStream } from '../../hooks/useWebSocket';
import { clsx } from 'clsx';

interface LiveTrace {
  traceId: string;
  service: string;
  operation: string;
  model: string;
  status: 'ok' | 'error';
  duration: number;
  timestamp: string;
}

interface LiveTraceStreamProps {
  modelId?: string;
  maxTraces?: number;
  onTraceClick?: (traceId: string) => void;
}

export default function LiveTraceStream({
  modelId,
  maxTraces = 50,
  onTraceClick,
}: LiveTraceStreamProps) {
  const [isPaused, setIsPaused] = useState(false);
  const [pausedTraces, setPausedTraces] = useState<LiveTrace[]>([]);

  const handleNewTrace = useCallback(
    (trace: unknown) => {
      if (isPaused) {
        setPausedTraces((prev) => [trace as LiveTrace, ...prev].slice(0, maxTraces));
      }
    },
    [isPaused, maxTraces]
  );

  const { traces: liveTraces, isConnected, clearTraces } = useTraceStream(
    modelId,
    handleNewTrace
  );

  const traces = isPaused ? pausedTraces : (liveTraces as LiveTrace[]);

  const handlePause = () => {
    if (!isPaused) {
      setPausedTraces(liveTraces as LiveTrace[]);
    }
    setIsPaused(!isPaused);
  };

  const handleClear = () => {
    clearTraces();
    setPausedTraces([]);
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-800">
        <div className="flex items-center gap-3">
          <h3 className="font-medium text-gray-900 dark:text-white">
            Live Trace Stream
          </h3>

          {/* Connection Status */}
          <div
            className={clsx(
              'flex items-center gap-1 rounded-full px-2 py-0.5 text-xs',
              isConnected
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
            )}
          >
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3" />
                Connected
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3" />
                Disconnected
              </>
            )}
          </div>

          {/* Paused indicator */}
          {isPaused && (
            <span className="rounded-full bg-yellow-100 px-2 py-0.5 text-xs text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400">
              Paused
            </span>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handlePause}
            className={clsx(
              'btn-ghost flex items-center gap-1 px-2 py-1',
              isPaused && 'text-green-600 dark:text-green-400'
            )}
            title={isPaused ? 'Resume' : 'Pause'}
          >
            {isPaused ? (
              <>
                <Play className="h-4 w-4" />
                <span className="text-sm">Resume</span>
              </>
            ) : (
              <>
                <Pause className="h-4 w-4" />
                <span className="text-sm">Pause</span>
              </>
            )}
          </button>

          <button
            onClick={handleClear}
            className="btn-ghost flex items-center gap-1 px-2 py-1 text-red-600 dark:text-red-400"
            title="Clear traces"
          >
            <Trash2 className="h-4 w-4" />
            <span className="text-sm">Clear</span>
          </button>
        </div>
      </div>

      {/* Trace List */}
      <div className="max-h-96 overflow-y-auto">
        {traces.length === 0 ? (
          <div className="flex h-32 flex-col items-center justify-center text-gray-500 dark:text-gray-400">
            {isConnected ? (
              <>
                <div className="mb-2 h-4 w-4 animate-pulse rounded-full bg-green-500" />
                <p>Waiting for traces...</p>
              </>
            ) : (
              <>
                <WifiOff className="mb-2 h-6 w-6" />
                <p>Not connected to stream</p>
              </>
            )}
          </div>
        ) : (
          <div className="divide-y divide-gray-100 dark:divide-gray-800">
            {traces.map((trace, index) => (
              <div
                key={`${trace.traceId}-${index}`}
                className={clsx(
                  'flex items-center gap-4 px-4 py-2 transition-colors',
                  'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50',
                  index === 0 && !isPaused && 'animate-pulse bg-pyflare-50/50 dark:bg-pyflare-900/10'
                )}
                onClick={() => onTraceClick?.(trace.traceId)}
              >
                {/* Status Icon */}
                {trace.status === 'ok' ? (
                  <CheckCircle className="h-4 w-4 flex-shrink-0 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 flex-shrink-0 text-red-500" />
                )}

                {/* Trace Info */}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/traces/${trace.traceId}`}
                      className="font-mono text-sm text-pyflare-600 hover:underline dark:text-pyflare-400"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {trace.traceId.substring(0, 12)}...
                    </Link>
                    <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                      {trace.model}
                    </span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                    <span>{trace.service}</span>
                    <span>·</span>
                    <span>{trace.operation}</span>
                  </div>
                </div>

                {/* Duration */}
                <div className="flex-shrink-0 text-right">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {trace.duration >= 1000
                      ? `${(trace.duration / 1000).toFixed(2)}s`
                      : `${trace.duration}ms`}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {formatDistanceToNow(new Date(trace.timestamp), { addSuffix: true })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-gray-200 px-4 py-2 dark:border-gray-800">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Showing {traces.length} of {maxTraces} max traces
          {modelId && <span> · Filtering by model: {modelId}</span>}
        </p>
      </div>
    </div>
  );
}
