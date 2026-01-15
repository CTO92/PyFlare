/**
 * Traces Page
 * Enhanced trace explorer with search, filters, and real-time streaming
 */

import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Filter, X, Download, GitCompare, Radio } from 'lucide-react';
import TraceSearch, { parseQuery, type TraceQuery } from '../components/traces/TraceSearch';
import TraceFilters, { defaultTraceFilters } from '../components/traces/TraceFilters';
import TraceList, { type TraceItem } from '../components/traces/TraceList';
import LiveTraceStream from '../components/traces/LiveTraceStream';
import { useTraces } from '../hooks/useTraces';

type ViewMode = 'list' | 'live';

// Mock saved queries - in production these would come from API/localStorage
const savedQueries = [
  { name: 'Errors Today', query: 'status:error time:last-24h' },
  { name: 'GPT-4 Slow', query: 'model:gpt-4 duration:>2000' },
  { name: 'Drift Issues', query: 'has:drift time:last-7d' },
];

// Mock recent queries - in production these would come from localStorage
const recentQueries = [
  'service:chat-api status:error',
  'model:claude-3 duration:>1000',
  'has:safety time:last-1h',
];

export default function Traces() {
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(true);
  const [activeFilters, setActiveFilters] = useState<Record<string, string[]>>({});
  const [selectedTraces, setSelectedTraces] = useState<string[]>([]);
  const [parsedQuery, setParsedQuery] = useState<TraceQuery>({});

  // Fetch traces
  const { traces: rawTraces, loading, error, pagination, refetch } = useTraces({
    modelId: parsedQuery.model_id,
    status: parsedQuery.status,
    page: 1,
    pageSize: 50,
  });

  // Transform API traces to TraceItem format
  const traces: TraceItem[] = useMemo(() => {
    return rawTraces.map((trace) => ({
      id: trace.traceId,
      traceId: trace.traceId,
      service: trace.modelId?.split('/')[0] || 'unknown',
      operation: 'inference',
      model: trace.modelId || 'unknown',
      status: trace.status,
      duration: trace.latencyMs,
      tokens: {
        input: trace.inputTokens,
        output: trace.outputTokens,
      },
      cost: trace.costMicros,
      timestamp: trace.startTime,
      hasDrift: (trace.driftScore ?? 0) > 0.3,
      hasSafetyIssue: (trace.toxicityScore ?? 0) > 0.5,
      evalScore: trace.evalScore,
    }));
  }, [rawTraces]);

  const handleSearch = (query: TraceQuery) => {
    setParsedQuery(query);
    refetch();
  };

  const handleFilterChange = (filters: Record<string, string[]>) => {
    setActiveFilters(filters);
    // Update parsed query based on filters
    const newQuery: TraceQuery = { ...parsedQuery };
    if (filters.status?.length) {
      newQuery.status = filters.status[0] as 'ok' | 'error';
    }
    if (filters.model?.length) {
      newQuery.model_id = filters.model[0];
    }
    setParsedQuery(newQuery);
    refetch();
  };

  const handleClearFilters = () => {
    setActiveFilters({});
    setParsedQuery({});
    refetch();
  };

  const handleTraceClick = (traceId: string) => {
    navigate(`/traces/${traceId}`);
  };

  const handleCompare = () => {
    if (selectedTraces.length === 2) {
      navigate(`/traces/compare?left=${selectedTraces[0]}&right=${selectedTraces[1]}`);
    }
  };

  const handleExport = () => {
    const data = traces.map((t) => ({
      traceId: t.traceId,
      service: t.service,
      model: t.model,
      status: t.status,
      duration: t.duration,
      inputTokens: t.tokens.input,
      outputTokens: t.tokens.output,
      cost: t.cost,
      timestamp: t.timestamp,
    }));

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `traces-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeFilterCount = Object.values(activeFilters).flat().length;

  return (
    <div className="flex h-full">
      {/* Filters Sidebar */}
      {showFilters && viewMode === 'list' && (
        <TraceFilters
          filters={defaultTraceFilters}
          activeFilters={activeFilters}
          onFilterChange={handleFilterChange}
          onClearAll={handleClearFilters}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Traces
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Explore and analyze your ML inference traces
          </p>
        </div>

        {/* Search and Actions Bar */}
        <div className="mb-6 flex flex-wrap items-center gap-4">
          {/* View Mode Toggle */}
          <div className="flex rounded-lg border border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setViewMode('list')}
              className={`flex items-center gap-2 px-4 py-2 text-sm ${
                viewMode === 'list'
                  ? 'bg-pyflare-50 text-pyflare-600 dark:bg-pyflare-900/20 dark:text-pyflare-400'
                  : 'text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800'
              }`}
            >
              List View
            </button>
            <button
              onClick={() => setViewMode('live')}
              className={`flex items-center gap-2 px-4 py-2 text-sm ${
                viewMode === 'live'
                  ? 'bg-pyflare-50 text-pyflare-600 dark:bg-pyflare-900/20 dark:text-pyflare-400'
                  : 'text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800'
              }`}
            >
              <Radio className="h-4 w-4" />
              Live Stream
            </button>
          </div>

          {/* Search */}
          {viewMode === 'list' && (
            <TraceSearch
              value={searchQuery}
              onChange={setSearchQuery}
              onSearch={handleSearch}
              savedQueries={savedQueries}
              recentQueries={recentQueries}
            />
          )}

          {/* Filter Toggle */}
          {viewMode === 'list' && (
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`btn-secondary flex items-center gap-2 ${
                showFilters ? 'bg-pyflare-50 text-pyflare-600 dark:bg-pyflare-900/20' : ''
              }`}
            >
              <Filter className="h-4 w-4" />
              Filters
              {activeFilterCount > 0 && (
                <span className="rounded-full bg-pyflare-500 px-2 py-0.5 text-xs text-white">
                  {activeFilterCount}
                </span>
              )}
            </button>
          )}

          {/* Actions */}
          {viewMode === 'list' && (
            <div className="flex items-center gap-2">
              {/* Compare Button */}
              {selectedTraces.length === 2 && (
                <button
                  onClick={handleCompare}
                  className="btn-secondary flex items-center gap-2"
                >
                  <GitCompare className="h-4 w-4" />
                  Compare
                </button>
              )}

              {/* Export Button */}
              <button
                onClick={handleExport}
                className="btn-secondary flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Export
              </button>
            </div>
          )}
        </div>

        {/* Active Filter Tags */}
        {viewMode === 'list' && activeFilterCount > 0 && (
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Active filters:
            </span>
            {Object.entries(activeFilters).map(([key, values]) =>
              values.map((value) => (
                <span
                  key={`${key}-${value}`}
                  className="inline-flex items-center gap-1 rounded-full bg-pyflare-100 px-3 py-1 text-sm text-pyflare-700 dark:bg-pyflare-900/30 dark:text-pyflare-400"
                >
                  {key}: {value}
                  <button
                    onClick={() => {
                      const newFilters = { ...activeFilters };
                      newFilters[key] = values.filter((v) => v !== value);
                      if (newFilters[key].length === 0) {
                        delete newFilters[key];
                      }
                      setActiveFilters(newFilters);
                    }}
                    className="hover:text-pyflare-900 dark:hover:text-pyflare-200"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))
            )}
            <button
              onClick={handleClearFilters}
              className="text-sm text-pyflare-600 hover:text-pyflare-700 dark:text-pyflare-400"
            >
              Clear all
            </button>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mb-4 rounded-lg bg-red-50 p-4 text-red-700 dark:bg-red-900/20 dark:text-red-400">
            {error}
          </div>
        )}

        {/* Content */}
        {viewMode === 'list' ? (
          <>
            <TraceList
              traces={traces}
              loading={loading}
              selectedTraceIds={selectedTraces}
              onSelectionChange={setSelectedTraces}
              onTraceClick={handleTraceClick}
            />

            {/* Pagination */}
            {!loading && traces.length > 0 && (
              <div className="mt-4 flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
                <span>
                  Showing {traces.length} of {pagination.total} traces
                </span>
                <div className="flex items-center gap-2">
                  <button
                    disabled={pagination.page <= 1}
                    className="btn-secondary disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <span>
                    Page {pagination.page} of {Math.ceil(pagination.total / pagination.pageSize)}
                  </span>
                  <button
                    disabled={pagination.page >= Math.ceil(pagination.total / pagination.pageSize)}
                    className="btn-secondary disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <LiveTraceStream
            maxTraces={100}
            onTraceClick={handleTraceClick}
          />
        )}
      </div>
    </div>
  );
}
