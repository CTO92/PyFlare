/**
 * TraceViewer Component
 * Detailed visualization of a single trace including spans, evaluations, and metrics
 *
 * SECURITY: All user-controlled content is sanitized to prevent XSS attacks.
 */

import React from 'react';
import { useTrace, useTraceTimeline } from '../hooks/useTraces';
import { formatCost, formatTokens } from '../hooks/useCosts';

// =============================================================================
// SECURITY: Sanitization Utilities
// =============================================================================

/**
 * SECURITY: Sanitize a string to prevent XSS attacks
 * Removes HTML tags and escapes special characters
 */
function sanitizeText(input: string | null | undefined, maxLength: number = 10000): string {
  if (!input) return '';

  // Truncate to max length to prevent DoS
  let text = input.length > maxLength ? input.substring(0, maxLength) + '...' : input;

  // Remove any HTML tags
  text = text.replace(/<[^>]*>/g, '');

  // Escape special HTML characters
  text = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

  return text;
}

/**
 * SECURITY: Validate and sanitize a CSS class name
 * Only allows alphanumeric characters and hyphens
 */
function sanitizeClassName(input: string | null | undefined): string {
  if (!input) return 'unknown';
  // Only allow alphanumeric, hyphens, and underscores
  const sanitized = input.replace(/[^a-zA-Z0-9_-]/g, '');
  return sanitized || 'unknown';
}

/**
 * SECURITY: Sanitize content for display in <pre> tags
 */
function sanitizePreContent(input: string | null | undefined, maxLength: number = 50000): string {
  if (!input) return '';

  // Truncate to prevent DoS
  let text = input.length > maxLength
    ? input.substring(0, maxLength) + '\n\n[Content truncated...]'
    : input;

  // Remove any script tags and event handlers
  text = text.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '[script removed]');
  text = text.replace(/on\w+\s*=/gi, 'data-removed=');

  return text;
}

interface TraceViewerProps {
  traceId: string;
  onClose?: () => void;
}

export function TraceViewer({ traceId, onClose }: TraceViewerProps) {
  const { trace, loading, error } = useTrace(traceId);
  const { timeline } = useTraceTimeline(traceId);

  if (loading) {
    return (
      <div className="trace-viewer">
        <div className="trace-viewer-loading">Loading trace...</div>
      </div>
    );
  }

  if (error || !trace) {
    return (
      <div className="trace-viewer">
        <div className="trace-viewer-error">
          {/* SECURITY: Sanitize error message to prevent XSS */}
          {sanitizeText(error) || 'Trace not found'}
        </div>
      </div>
    );
  }

  return (
    <div className="trace-viewer">
      {/* Header */}
      <div className="trace-viewer-header">
        <div className="trace-viewer-title">
          <h2>Trace {trace.traceId.substring(0, 8)}...</h2>
          {/* SECURITY: Sanitize status for CSS class name */}
          <span className={`status-badge status-${sanitizeClassName(trace.status)}`}>
            {sanitizeText(trace.status?.toUpperCase())}
          </span>
        </div>
        {onClose && (
          <button className="close-button" onClick={onClose}>
            &times;
          </button>
        )}
      </div>

      {/* Overview Cards */}
      <div className="trace-viewer-overview">
        <div className="metric-card">
          <label>Latency</label>
          <value>{trace.latencyMs}ms</value>
        </div>
        <div className="metric-card">
          <label>Tokens</label>
          <value>{formatTokens(trace.totalTokens)}</value>
        </div>
        <div className="metric-card">
          <label>Cost</label>
          <value>{formatCost(trace.costMicros)}</value>
        </div>
        <div className="metric-card">
          <label>Model</label>
          <value>{trace.modelId}</value>
        </div>
      </div>

      {/* Error Details */}
      {trace.status === 'error' && (
        <div className="trace-viewer-error-details">
          <h3>Error Details</h3>
          {/* SECURITY: Sanitize error details to prevent XSS */}
          <div className="error-type">{sanitizeText(trace.errorType)}</div>
          <div className="error-message">{sanitizeText(trace.errorMessage)}</div>
        </div>
      )}

      {/* Evaluation Scores */}
      <div className="trace-viewer-evaluations">
        <h3>Evaluations</h3>
        <div className="evaluation-scores">
          {trace.evalScore !== undefined && (
            <div className="score-item">
              <label>Quality Score</label>
              <div className="score-bar">
                <div
                  className="score-fill"
                  style={{ width: `${trace.evalScore * 100}%` }}
                />
              </div>
              <span>{(trace.evalScore * 100).toFixed(0)}%</span>
            </div>
          )}
          {trace.driftScore !== undefined && (
            <div className="score-item">
              <label>Drift Score</label>
              <div className="score-bar">
                <div
                  className={`score-fill ${trace.driftScore > 0.3 ? 'warning' : ''}`}
                  style={{ width: `${trace.driftScore * 100}%` }}
                />
              </div>
              <span>{(trace.driftScore * 100).toFixed(0)}%</span>
            </div>
          )}
          {trace.toxicityScore !== undefined && (
            <div className="score-item">
              <label>Toxicity Score</label>
              <div className="score-bar">
                <div
                  className={`score-fill ${trace.toxicityScore > 0.5 ? 'danger' : ''}`}
                  style={{ width: `${trace.toxicityScore * 100}%` }}
                />
              </div>
              <span>{(trace.toxicityScore * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      </div>

      {/* Token Usage */}
      <div className="trace-viewer-tokens">
        <h3>Token Usage</h3>
        <div className="token-breakdown">
          <div className="token-item">
            <label>Input</label>
            <value>{formatTokens(trace.inputTokens)}</value>
          </div>
          <div className="token-item">
            <label>Output</label>
            <value>{formatTokens(trace.outputTokens)}</value>
          </div>
          <div className="token-item total">
            <label>Total</label>
            <value>{formatTokens(trace.totalTokens)}</value>
          </div>
        </div>
      </div>

      {/* Timeline */}
      {timeline && timeline.spans.length > 0 && (
        <div className="trace-viewer-timeline">
          <h3>Span Timeline</h3>
          <div className="span-list">
            {timeline.spans.map((span, index) => (
              <div key={span.spanId} className="span-item">
                <div className="span-index">{index + 1}</div>
                <div className="span-details">
                  <div className="span-name">{span.spanId.substring(0, 8)}...</div>
                  <div className="span-duration">{span.latencyMs}ms</div>
                </div>
                <div
                  className="span-bar"
                  style={{
                    width: `${(span.latencyMs / trace.latencyMs) * 100}%`,
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input/Output */}
      <div className="trace-viewer-io">
        <div className="io-section">
          <h3>Input</h3>
          {/* SECURITY: Sanitize input content to prevent XSS */}
          <pre className="io-content">{sanitizePreContent(trace.input) || 'N/A'}</pre>
        </div>
        <div className="io-section">
          <h3>Output</h3>
          {/* SECURITY: Sanitize output content to prevent XSS */}
          <pre className="io-content">{sanitizePreContent(trace.output) || 'N/A'}</pre>
        </div>
      </div>

      {/* Metadata */}
      <div className="trace-viewer-metadata">
        <h3>Metadata</h3>
        <table className="metadata-table">
          <tbody>
            <tr>
              <td>Trace ID</td>
              <td><code>{trace.traceId}</code></td>
            </tr>
            <tr>
              <td>Span ID</td>
              <td><code>{trace.spanId}</code></td>
            </tr>
            <tr>
              <td>User ID</td>
              <td>{trace.userId || 'N/A'}</td>
            </tr>
            <tr>
              <td>Feature ID</td>
              <td>{trace.featureId || 'N/A'}</td>
            </tr>
            <tr>
              <td>Start Time</td>
              <td>{new Date(trace.startTime).toLocaleString()}</td>
            </tr>
            <tr>
              <td>End Time</td>
              <td>{new Date(trace.endTime).toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default TraceViewer;
