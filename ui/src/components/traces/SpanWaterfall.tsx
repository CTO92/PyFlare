/**
 * Span Waterfall Component
 * Hierarchical visualization of spans in a trace
 */

import { useState, useMemo } from 'react';
import { ChevronRight, ChevronDown, AlertTriangle, Clock, Zap } from 'lucide-react';
import { clsx } from 'clsx';

export interface Span {
  spanId: string;
  parentSpanId?: string;
  operationName: string;
  serviceName: string;
  startTime: number; // Unix timestamp in ms
  duration: number; // Duration in ms
  status: 'ok' | 'error';
  attributes?: Record<string, string>;
  events?: SpanEvent[];
}

export interface SpanEvent {
  name: string;
  timestamp: number;
  attributes?: Record<string, string>;
}

interface SpanWaterfallProps {
  spans: Span[];
  traceStart: number;
  traceDuration: number;
  selectedSpanId?: string;
  onSpanSelect?: (spanId: string) => void;
  showCriticalPath?: boolean;
}

interface SpanNode extends Span {
  children: SpanNode[];
  depth: number;
  isCriticalPath: boolean;
}

// Color palette for different services
const SERVICE_COLORS = [
  'bg-blue-500',
  'bg-green-500',
  'bg-purple-500',
  'bg-yellow-500',
  'bg-pink-500',
  'bg-indigo-500',
  'bg-teal-500',
  'bg-orange-500',
];

function buildSpanTree(spans: Span[], criticalPath: Set<string>): SpanNode[] {
  const spanMap = new Map<string, SpanNode>();
  const roots: SpanNode[] = [];

  // Create nodes
  spans.forEach((span) => {
    spanMap.set(span.spanId, {
      ...span,
      children: [],
      depth: 0,
      isCriticalPath: criticalPath.has(span.spanId),
    });
  });

  // Build tree
  spans.forEach((span) => {
    const node = spanMap.get(span.spanId)!;
    if (span.parentSpanId && spanMap.has(span.parentSpanId)) {
      const parent = spanMap.get(span.parentSpanId)!;
      parent.children.push(node);
      node.depth = parent.depth + 1;
    } else {
      roots.push(node);
    }
  });

  // Sort children by start time
  const sortChildren = (nodes: SpanNode[]) => {
    nodes.sort((a, b) => a.startTime - b.startTime);
    nodes.forEach((node) => sortChildren(node.children));
  };
  sortChildren(roots);

  return roots;
}

function findCriticalPath(spans: Span[]): Set<string> {
  const criticalPath = new Set<string>();

  // Find the longest duration path
  const findPath = (spanId: string, spans: Span[]): number => {
    const span = spans.find((s) => s.spanId === spanId);
    if (!span) return 0;

    const children = spans.filter((s) => s.parentSpanId === spanId);
    if (children.length === 0) return span.duration;

    let maxChildDuration = 0;
    let maxChild: Span | null = null;

    children.forEach((child) => {
      const childDuration = findPath(child.spanId, spans);
      if (childDuration > maxChildDuration) {
        maxChildDuration = childDuration;
        maxChild = child;
      }
    });

    if (maxChild) {
      criticalPath.add(maxChild.spanId);
    }

    return span.duration;
  };

  const roots = spans.filter((s) => !s.parentSpanId);
  roots.forEach((root) => {
    criticalPath.add(root.spanId);
    findPath(root.spanId, spans);
  });

  return criticalPath;
}

function SpanRow({
  node,
  traceStart,
  traceDuration,
  selectedSpanId,
  onSpanSelect,
  serviceColorMap,
  isExpanded,
  onToggle,
}: {
  node: SpanNode;
  traceStart: number;
  traceDuration: number;
  selectedSpanId?: string;
  onSpanSelect?: (spanId: string) => void;
  serviceColorMap: Map<string, string>;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const leftPercent = ((node.startTime - traceStart) / traceDuration) * 100;
  const widthPercent = (node.duration / traceDuration) * 100;
  const hasChildren = node.children.length > 0;
  const isSelected = selectedSpanId === node.spanId;
  const barColor = serviceColorMap.get(node.serviceName) || 'bg-gray-500';

  return (
    <div
      className={clsx(
        'group flex items-center border-b border-gray-100 py-1 dark:border-gray-800',
        isSelected && 'bg-pyflare-50 dark:bg-pyflare-900/20',
        'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50'
      )}
      onClick={() => onSpanSelect?.(node.spanId)}
    >
      {/* Left side: span info */}
      <div
        className="flex w-64 flex-shrink-0 items-center gap-1 overflow-hidden px-2"
        style={{ paddingLeft: `${node.depth * 16 + 8}px` }}
      >
        {/* Expand/collapse button */}
        {hasChildren ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            className="flex-shrink-0 rounded p-0.5 hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-500" />
            )}
          </button>
        ) : (
          <div className="w-5" />
        )}

        {/* Status indicator */}
        {node.status === 'error' && (
          <AlertTriangle className="h-4 w-4 flex-shrink-0 text-red-500" />
        )}

        {/* Critical path indicator */}
        {node.isCriticalPath && (
          <Zap className="h-4 w-4 flex-shrink-0 text-yellow-500" title="Critical path" />
        )}

        {/* Operation name */}
        <span className="truncate text-sm font-medium text-gray-900 dark:text-white">
          {node.operationName}
        </span>
      </div>

      {/* Right side: timeline bar */}
      <div className="relative flex-1">
        <div className="absolute inset-0 flex items-center">
          <div
            className={clsx(
              'relative h-6 rounded',
              barColor,
              node.status === 'error' && 'opacity-70',
              node.isCriticalPath && 'ring-2 ring-yellow-400'
            )}
            style={{
              left: `${leftPercent}%`,
              width: `${Math.max(widthPercent, 0.5)}%`,
            }}
          >
            {/* Duration label */}
            <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
              {node.duration >= 1000 ? `${(node.duration / 1000).toFixed(1)}s` : `${node.duration}ms`}
            </span>
          </div>
        </div>
      </div>

      {/* Service name */}
      <div className="w-32 flex-shrink-0 px-2">
        <span className="truncate text-xs text-gray-500 dark:text-gray-400">
          {node.serviceName}
        </span>
      </div>
    </div>
  );
}

export default function SpanWaterfall({
  spans,
  traceStart,
  traceDuration,
  selectedSpanId,
  onSpanSelect,
  showCriticalPath = true,
}: SpanWaterfallProps) {
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set(spans.map((s) => s.spanId)));

  const criticalPath = useMemo(
    () => (showCriticalPath ? findCriticalPath(spans) : new Set<string>()),
    [spans, showCriticalPath]
  );

  const spanTree = useMemo(() => buildSpanTree(spans, criticalPath), [spans, criticalPath]);

  // Generate service color map
  const serviceColorMap = useMemo(() => {
    const services = [...new Set(spans.map((s) => s.serviceName))];
    const map = new Map<string, string>();
    services.forEach((service, i) => {
      map.set(service, SERVICE_COLORS[i % SERVICE_COLORS.length]);
    });
    return map;
  }, [spans]);

  const toggleExpand = (spanId: string) => {
    const newExpanded = new Set(expandedSpans);
    if (newExpanded.has(spanId)) {
      newExpanded.delete(spanId);
    } else {
      newExpanded.add(spanId);
    }
    setExpandedSpans(newExpanded);
  };

  const expandAll = () => setExpandedSpans(new Set(spans.map((s) => s.spanId)));
  const collapseAll = () => setExpandedSpans(new Set());

  // Render spans recursively
  const renderSpan = (node: SpanNode): React.ReactNode[] => {
    const isExpanded = expandedSpans.has(node.spanId);
    const result = [
      <SpanRow
        key={node.spanId}
        node={node}
        traceStart={traceStart}
        traceDuration={traceDuration}
        selectedSpanId={selectedSpanId}
        onSpanSelect={onSpanSelect}
        serviceColorMap={serviceColorMap}
        isExpanded={isExpanded}
        onToggle={() => toggleExpand(node.spanId)}
      />,
    ];

    if (isExpanded) {
      node.children.forEach((child) => {
        result.push(...renderSpan(child));
      });
    }

    return result;
  };

  // Generate time ruler marks
  const timeMarks = useMemo(() => {
    const marks: { percent: number; label: string }[] = [];
    const intervals = [0, 25, 50, 75, 100];
    intervals.forEach((percent) => {
      const ms = (traceDuration * percent) / 100;
      marks.push({
        percent,
        label: ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`,
      });
    });
    return marks;
  }, [traceDuration]);

  if (spans.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-gray-500 dark:text-gray-400">
        <Clock className="mr-2 h-5 w-5" />
        No spans to display
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2 dark:border-gray-800">
        <div className="flex items-center gap-4">
          <h3 className="font-medium text-gray-900 dark:text-white">
            Span Timeline
          </h3>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {spans.length} spans, {traceDuration >= 1000 ? `${(traceDuration / 1000).toFixed(2)}s` : `${traceDuration}ms`} total
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={expandAll}
            className="text-xs text-pyflare-600 hover:text-pyflare-700 dark:text-pyflare-400"
          >
            Expand all
          </button>
          <span className="text-gray-300 dark:text-gray-600">|</span>
          <button
            onClick={collapseAll}
            className="text-xs text-pyflare-600 hover:text-pyflare-700 dark:text-pyflare-400"
          >
            Collapse all
          </button>
        </div>
      </div>

      {/* Time ruler */}
      <div className="flex border-b border-gray-200 dark:border-gray-800">
        <div className="w-64 flex-shrink-0" />
        <div className="relative flex-1 px-2 py-1">
          {timeMarks.map((mark) => (
            <div
              key={mark.percent}
              className="absolute text-xs text-gray-400"
              style={{ left: `${mark.percent}%`, transform: 'translateX(-50%)' }}
            >
              {mark.label}
            </div>
          ))}
        </div>
        <div className="w-32 flex-shrink-0" />
      </div>

      {/* Span rows */}
      <div className="max-h-96 overflow-y-auto">
        {spanTree.flatMap((node) => renderSpan(node))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 border-t border-gray-200 px-4 py-2 dark:border-gray-800">
        {[...serviceColorMap.entries()].map(([service, color]) => (
          <div key={service} className="flex items-center gap-1">
            <div className={clsx('h-3 w-3 rounded', color)} />
            <span className="text-xs text-gray-600 dark:text-gray-400">{service}</span>
          </div>
        ))}
        {showCriticalPath && (
          <div className="flex items-center gap-1">
            <Zap className="h-3 w-3 text-yellow-500" />
            <span className="text-xs text-gray-600 dark:text-gray-400">Critical path</span>
          </div>
        )}
      </div>
    </div>
  );
}
