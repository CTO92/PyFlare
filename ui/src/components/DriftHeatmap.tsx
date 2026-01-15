/**
 * DriftHeatmap Component
 * Visualizes drift scores across features and models
 */

import React, { useMemo } from 'react';
import { useDriftHeatmap } from '../hooks/useDrift';

interface DriftHeatmapProps {
  onCellClick?: (feature: string, model: string, value: number) => void;
}

export function DriftHeatmap({ onCellClick }: DriftHeatmapProps) {
  const { heatmap, loading, error } = useDriftHeatmap();

  // Color scale for drift values
  const getColor = (value: number): string => {
    if (value < 0.1) return 'var(--color-success, #22c55e)';
    if (value < 0.25) return 'var(--color-warning, #eab308)';
    if (value < 0.5) return 'var(--color-orange, #f97316)';
    return 'var(--color-danger, #ef4444)';
  };

  const getBackgroundColor = (value: number): string => {
    const intensity = Math.min(value * 2, 1); // Scale for visibility
    if (value < 0.1) return `rgba(34, 197, 94, ${intensity * 0.3})`;
    if (value < 0.25) return `rgba(234, 179, 8, ${intensity * 0.5})`;
    if (value < 0.5) return `rgba(249, 115, 22, ${intensity * 0.7})`;
    return `rgba(239, 68, 68, ${intensity * 0.9})`;
  };

  // Find max value for scale reference
  const maxValue = useMemo(() => {
    if (!heatmap) return 1;
    return Math.max(...heatmap.matrix.flat(), 0.01);
  }, [heatmap]);

  if (loading) {
    return (
      <div className="drift-heatmap">
        <div className="drift-heatmap-loading">Loading heatmap data...</div>
      </div>
    );
  }

  if (error || !heatmap) {
    return (
      <div className="drift-heatmap">
        <div className="drift-heatmap-error">{error || 'No data available'}</div>
      </div>
    );
  }

  const { features, models, matrix } = heatmap;

  if (features.length === 0 || models.length === 0) {
    return (
      <div className="drift-heatmap">
        <div className="drift-heatmap-empty">No drift data to display</div>
      </div>
    );
  }

  return (
    <div className="drift-heatmap">
      <div className="drift-heatmap-header">
        <h3>Feature Drift Heatmap</h3>
        <div className="drift-heatmap-legend">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: getColor(0.05) }} />
            <span>Low (&lt;0.1)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: getColor(0.15) }} />
            <span>Moderate (0.1-0.25)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: getColor(0.35) }} />
            <span>High (0.25-0.5)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: getColor(0.75) }} />
            <span>Critical (&gt;0.5)</span>
          </div>
        </div>
      </div>

      <div className="drift-heatmap-container">
        <table className="drift-heatmap-table">
          <thead>
            <tr>
              <th></th>
              {models.map((model) => (
                <th key={model} className="model-header">
                  <span className="model-name">{model}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feature, featureIdx) => (
              <tr key={feature}>
                <td className="feature-label">
                  <span className="feature-name">{feature}</span>
                </td>
                {models.map((model, modelIdx) => {
                  const value = matrix[featureIdx]?.[modelIdx] ?? 0;
                  return (
                    <td
                      key={`${feature}-${model}`}
                      className="heatmap-cell"
                      style={{
                        backgroundColor: getBackgroundColor(value),
                        cursor: onCellClick ? 'pointer' : 'default',
                      }}
                      onClick={() => onCellClick?.(feature, model, value)}
                      title={`${feature} - ${model}: ${(value * 100).toFixed(1)}%`}
                    >
                      <span className="cell-value">{(value * 100).toFixed(0)}%</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="drift-heatmap-summary">
        <div className="summary-item">
          <label>Features Monitored</label>
          <value>{features.length}</value>
        </div>
        <div className="summary-item">
          <label>Models Monitored</label>
          <value>{models.length}</value>
        </div>
        <div className="summary-item">
          <label>Max Drift Score</label>
          <value style={{ color: getColor(maxValue) }}>
            {(maxValue * 100).toFixed(1)}%
          </value>
        </div>
      </div>
    </div>
  );
}

export default DriftHeatmap;
