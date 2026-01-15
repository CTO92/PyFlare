/**
 * Trace Search Component
 * Advanced search with query language support
 */

import { useState, useRef, useEffect } from 'react';
import { Search, X, History, Star, HelpCircle } from 'lucide-react';

export interface TraceQuery {
  service?: string;
  model_id?: string;
  status?: 'ok' | 'error';
  duration_min?: number;
  duration_max?: number;
  time_range?: string;
  has_drift?: boolean;
  has_safety_issues?: boolean;
  user_id?: string;
  text?: string;
}

interface TraceSearchProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: (query: TraceQuery) => void;
  savedQueries?: { name: string; query: string }[];
  recentQueries?: string[];
}

const QUERY_SYNTAX_HELP = [
  { example: 'service:my-service', description: 'Filter by service name' },
  { example: 'model:gpt-4', description: 'Filter by model ID' },
  { example: 'status:error', description: 'Filter by status (ok, error)' },
  { example: 'duration:>1000', description: 'Duration greater than 1000ms' },
  { example: 'duration:<500', description: 'Duration less than 500ms' },
  { example: 'time:last-1h', description: 'Time range (last-1h, last-24h, last-7d)' },
  { example: 'has:drift', description: 'Has drift detected' },
  { example: 'has:safety', description: 'Has safety issues' },
  { example: 'user:user123', description: 'Filter by user ID' },
];

/**
 * Parse query string into structured query object
 */
function parseQuery(queryString: string): TraceQuery {
  const query: TraceQuery = {};
  const parts = queryString.match(/(?:[^\s"]+|"[^"]*")+/g) || [];
  const textParts: string[] = [];

  for (const part of parts) {
    const [key, ...valueParts] = part.split(':');
    const value = valueParts.join(':').replace(/^"|"$/g, '');

    if (!value) {
      textParts.push(key);
      continue;
    }

    switch (key.toLowerCase()) {
      case 'service':
        query.service = value;
        break;
      case 'model':
        query.model_id = value;
        break;
      case 'status':
        query.status = value as 'ok' | 'error';
        break;
      case 'duration':
        if (value.startsWith('>')) {
          query.duration_min = parseInt(value.slice(1), 10);
        } else if (value.startsWith('<')) {
          query.duration_max = parseInt(value.slice(1), 10);
        }
        break;
      case 'time':
        query.time_range = value;
        break;
      case 'has':
        if (value === 'drift') query.has_drift = true;
        if (value === 'safety') query.has_safety_issues = true;
        break;
      case 'user':
        query.user_id = value;
        break;
      default:
        textParts.push(part);
    }
  }

  if (textParts.length > 0) {
    query.text = textParts.join(' ');
  }

  return query;
}

export default function TraceSearch({
  value,
  onChange,
  onSearch,
  savedQueries = [],
  recentQueries = [],
}: TraceSearchProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const query = parseQuery(value);
    onSearch(query);
    setShowDropdown(false);
  };

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        !inputRef.current?.contains(event.target as Node)
      ) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Focus search on '/' key
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      // Close on Escape
      if (e.key === 'Escape') {
        setShowDropdown(false);
        setShowHelp(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const selectQuery = (query: string) => {
    onChange(query);
    onSearch(parseQuery(query));
    setShowDropdown(false);
  };

  return (
    <div className="relative flex-1">
      <form onSubmit={handleSubmit} className="relative">
        <Search className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400" />
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setShowDropdown(true)}
          placeholder="Search traces... (service:name model:gpt-4 status:error)"
          className="input h-10 w-full pl-10 pr-20"
          autoComplete="off"
        />
        <div className="absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-1">
          {value && (
            <button
              type="button"
              onClick={() => onChange('')}
              className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800"
            >
              <X className="h-4 w-4" />
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowHelp(!showHelp)}
            className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800"
            title="Search syntax help"
          >
            <HelpCircle className="h-4 w-4" />
          </button>
        </div>
      </form>

      {/* Dropdown */}
      {showDropdown && (savedQueries.length > 0 || recentQueries.length > 0) && (
        <div
          ref={dropdownRef}
          className="absolute left-0 right-0 top-full z-50 mt-1 rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800"
        >
          {/* Saved Queries */}
          {savedQueries.length > 0 && (
            <div className="border-b border-gray-200 p-2 dark:border-gray-700">
              <p className="px-2 py-1 text-xs font-medium text-gray-500 dark:text-gray-400">
                Saved Queries
              </p>
              {savedQueries.map((q, i) => (
                <button
                  key={i}
                  onClick={() => selectQuery(q.query)}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <Star className="h-4 w-4 text-yellow-500" />
                  <span className="font-medium text-gray-900 dark:text-white">
                    {q.name}
                  </span>
                  <span className="truncate text-gray-500 dark:text-gray-400">
                    {q.query}
                  </span>
                </button>
              ))}
            </div>
          )}

          {/* Recent Queries */}
          {recentQueries.length > 0 && (
            <div className="p-2">
              <p className="px-2 py-1 text-xs font-medium text-gray-500 dark:text-gray-400">
                Recent Searches
              </p>
              {recentQueries.slice(0, 5).map((q, i) => (
                <button
                  key={i}
                  onClick={() => selectQuery(q)}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <History className="h-4 w-4 text-gray-400" />
                  <span className="truncate text-gray-700 dark:text-gray-300">{q}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Help Popup */}
      {showHelp && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 rounded-lg border border-gray-200 bg-white p-4 shadow-lg dark:border-gray-700 dark:bg-gray-800">
          <h4 className="mb-3 font-medium text-gray-900 dark:text-white">
            Search Syntax
          </h4>
          <div className="space-y-2">
            {QUERY_SYNTAX_HELP.map((item, i) => (
              <div key={i} className="flex items-center gap-3">
                <code className="rounded bg-gray-100 px-2 py-0.5 text-sm dark:bg-gray-700">
                  {item.example}
                </code>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {item.description}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            Combine multiple filters: <code className="rounded bg-gray-100 px-1 dark:bg-gray-700">service:api model:gpt-4 status:error</code>
          </p>
        </div>
      )}
    </div>
  );
}

export { parseQuery };
