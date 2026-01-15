/**
 * Trace Filters Component
 * Sidebar with filterable facets
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, X } from 'lucide-react';

interface FilterOption {
  value: string;
  label: string;
  count?: number;
}

interface FilterSection {
  id: string;
  label: string;
  options: FilterOption[];
  multiSelect?: boolean;
}

interface ActiveFilters {
  [key: string]: string[];
}

interface TraceFiltersProps {
  filters: FilterSection[];
  activeFilters: ActiveFilters;
  onFilterChange: (filters: ActiveFilters) => void;
  onClearAll: () => void;
}

export default function TraceFilters({
  filters,
  activeFilters,
  onFilterChange,
  onClearAll,
}: TraceFiltersProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(filters.map((f) => f.id))
  );

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const toggleFilter = (sectionId: string, value: string, multiSelect = true) => {
    const currentValues = activeFilters[sectionId] || [];
    let newValues: string[];

    if (multiSelect) {
      if (currentValues.includes(value)) {
        newValues = currentValues.filter((v) => v !== value);
      } else {
        newValues = [...currentValues, value];
      }
    } else {
      newValues = currentValues.includes(value) ? [] : [value];
    }

    onFilterChange({
      ...activeFilters,
      [sectionId]: newValues,
    });
  };

  const activeFilterCount = Object.values(activeFilters).flat().length;

  return (
    <div className="w-64 flex-shrink-0 border-r border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-800">
        <h3 className="font-medium text-gray-900 dark:text-white">Filters</h3>
        {activeFilterCount > 0 && (
          <button
            onClick={onClearAll}
            className="flex items-center gap-1 text-xs text-pyflare-600 hover:text-pyflare-700 dark:text-pyflare-400"
          >
            <X className="h-3 w-3" />
            Clear all ({activeFilterCount})
          </button>
        )}
      </div>

      {/* Filter Sections */}
      <div className="overflow-y-auto p-2">
        {filters.map((section) => (
          <div key={section.id} className="mb-2">
            {/* Section Header */}
            <button
              onClick={() => toggleSection(section.id)}
              className="flex w-full items-center justify-between rounded px-2 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {section.label}
              </span>
              <div className="flex items-center gap-2">
                {activeFilters[section.id]?.length > 0 && (
                  <span className="rounded-full bg-pyflare-100 px-2 py-0.5 text-xs text-pyflare-700 dark:bg-pyflare-900/30 dark:text-pyflare-400">
                    {activeFilters[section.id].length}
                  </span>
                )}
                {expandedSections.has(section.id) ? (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-gray-500" />
                )}
              </div>
            </button>

            {/* Section Options */}
            {expandedSections.has(section.id) && (
              <div className="ml-2 mt-1 space-y-1">
                {section.options.map((option) => {
                  const isSelected = activeFilters[section.id]?.includes(option.value);

                  return (
                    <label
                      key={option.value}
                      className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 hover:bg-gray-100 dark:hover:bg-gray-800"
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleFilter(section.id, option.value, section.multiSelect)}
                        className="h-4 w-4 rounded border-gray-300 text-pyflare-600 focus:ring-pyflare-500"
                      />
                      <span className="flex-1 text-sm text-gray-700 dark:text-gray-300">
                        {option.label}
                      </span>
                      {option.count !== undefined && (
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {option.count}
                        </span>
                      )}
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Time Range Filter */}
      <div className="border-t border-gray-200 p-4 dark:border-gray-800">
        <h4 className="mb-2 text-sm font-medium text-gray-900 dark:text-white">
          Time Range
        </h4>
        <select
          value={activeFilters.timeRange?.[0] || 'last-1h'}
          onChange={(e) =>
            onFilterChange({ ...activeFilters, timeRange: [e.target.value] })
          }
          className="input w-full text-sm"
        >
          <option value="last-15m">Last 15 minutes</option>
          <option value="last-1h">Last hour</option>
          <option value="last-6h">Last 6 hours</option>
          <option value="last-24h">Last 24 hours</option>
          <option value="last-7d">Last 7 days</option>
          <option value="last-30d">Last 30 days</option>
          <option value="custom">Custom range</option>
        </select>
      </div>
    </div>
  );
}

// Default filter configuration
export const defaultTraceFilters: FilterSection[] = [
  {
    id: 'status',
    label: 'Status',
    options: [
      { value: 'ok', label: 'Success', count: 0 },
      { value: 'error', label: 'Error', count: 0 },
    ],
  },
  {
    id: 'model',
    label: 'Model',
    multiSelect: true,
    options: [
      { value: 'gpt-4', label: 'GPT-4', count: 0 },
      { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo', count: 0 },
      { value: 'claude-3-opus', label: 'Claude 3 Opus', count: 0 },
      { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet', count: 0 },
    ],
  },
  {
    id: 'service',
    label: 'Service',
    multiSelect: true,
    options: [],
  },
  {
    id: 'duration',
    label: 'Duration',
    options: [
      { value: 'fast', label: '< 500ms', count: 0 },
      { value: 'normal', label: '500ms - 2s', count: 0 },
      { value: 'slow', label: '2s - 10s', count: 0 },
      { value: 'very_slow', label: '> 10s', count: 0 },
    ],
  },
  {
    id: 'flags',
    label: 'Flags',
    multiSelect: true,
    options: [
      { value: 'has_drift', label: 'Has Drift', count: 0 },
      { value: 'has_safety', label: 'Safety Issues', count: 0 },
      { value: 'has_eval', label: 'Has Evaluation', count: 0 },
    ],
  },
];
