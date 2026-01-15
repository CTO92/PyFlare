/**
 * RCAExplorer Component
 * Root Cause Analysis exploration interface
 */

import React, { useState } from 'react';
import { rcaApi, RCAReport, Pattern, FailureCluster, ProblematicSlice } from '../services/api';

interface RCAExplorerProps {
  modelId?: string;
  onPatternClick?: (pattern: Pattern) => void;
  onClusterClick?: (cluster: FailureCluster) => void;
  onSliceClick?: (slice: ProblematicSlice) => void;
}

export function RCAExplorer({
  modelId,
  onPatternClick,
  onClusterClick,
  onSliceClick,
}: RCAExplorerProps) {
  const [activeTab, setActiveTab] = useState<'patterns' | 'clusters' | 'slices'>('patterns');
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [clusters, setClusters] = useState<FailureCluster[]>([]);
  const [slices, setSlices] = useState<ProblematicSlice[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisRunning, setAnalysisRunning] = useState(false);

  const runAnalysis = async () => {
    setAnalysisRunning(true);
    setError(null);
    try {
      await rcaApi.runAnalysis({ modelId });
      // Refresh data after analysis
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalysisRunning(false);
    }
  };

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [patternsRes, clustersRes, slicesRes] = await Promise.all([
        rcaApi.getPatterns({ modelId }),
        rcaApi.getClusters({ modelId }),
        rcaApi.getSlices({ modelId }),
      ]);
      setPatterns(patternsRes.patterns);
      setClusters(clustersRes.clusters);
      setSlices(slicesRes.slices);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    loadData();
  }, [modelId]);

  const getSeverityColor = (severity: number) => {
    if (severity > 0.7) return 'var(--color-danger, #ef4444)';
    if (severity > 0.4) return 'var(--color-warning, #eab308)';
    return 'var(--color-success, #22c55e)';
  };

  return (
    <div className="rca-explorer">
      {/* Header */}
      <div className="rca-header">
        <div className="rca-title">
          <h2>Root Cause Analysis</h2>
          {modelId && <span className="model-badge">{modelId}</span>}
        </div>
        <button
          className="run-analysis-button"
          onClick={runAnalysis}
          disabled={analysisRunning}
        >
          {analysisRunning ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      {/* Tabs */}
      <div className="rca-tabs">
        <button
          className={activeTab === 'patterns' ? 'active' : ''}
          onClick={() => setActiveTab('patterns')}
        >
          Patterns ({patterns.length})
        </button>
        <button
          className={activeTab === 'clusters' ? 'active' : ''}
          onClick={() => setActiveTab('clusters')}
        >
          Failure Clusters ({clusters.length})
        </button>
        <button
          className={activeTab === 'slices' ? 'active' : ''}
          onClick={() => setActiveTab('slices')}
        >
          Problematic Slices ({slices.length})
        </button>
      </div>

      {/* Content */}
      {error && <div className="rca-error">{error}</div>}
      {loading && <div className="rca-loading">Loading...</div>}

      {!loading && !error && (
        <div className="rca-content">
          {/* Patterns Tab */}
          {activeTab === 'patterns' && (
            <div className="patterns-list">
              {patterns.length === 0 ? (
                <div className="empty-state">No patterns detected</div>
              ) : (
                patterns.map((pattern) => (
                  <div
                    key={pattern.id}
                    className="pattern-card"
                    onClick={() => onPatternClick?.(pattern)}
                  >
                    <div className="pattern-header">
                      <span className="pattern-type">{pattern.type}</span>
                      <span
                        className="pattern-severity"
                        style={{ color: getSeverityColor(pattern.severity) }}
                      >
                        Severity: {(pattern.severity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="pattern-description">{pattern.description}</div>
                    <div className="pattern-stats">
                      <span>{pattern.affectedTraces} traces affected</span>
                    </div>
                    {pattern.suggestedActions.length > 0 && (
                      <div className="pattern-actions">
                        <h4>Suggested Actions:</h4>
                        <ul>
                          {pattern.suggestedActions.slice(0, 3).map((action, idx) => (
                            <li key={idx}>{action}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Clusters Tab */}
          {activeTab === 'clusters' && (
            <div className="clusters-list">
              {clusters.length === 0 ? (
                <div className="empty-state">No failure clusters detected</div>
              ) : (
                clusters.map((cluster) => (
                  <div
                    key={cluster.id}
                    className="cluster-card"
                    onClick={() => onClusterClick?.(cluster)}
                  >
                    <div className="cluster-header">
                      <span className="cluster-name">{cluster.name}</span>
                      <span className="cluster-size">{cluster.size} failures</span>
                    </div>
                    <div className="cluster-error">
                      <strong>Representative Error:</strong>
                      <code>{cluster.representativeError}</code>
                    </div>
                    {cluster.commonKeywords.length > 0 && (
                      <div className="cluster-keywords">
                        {cluster.commonKeywords.slice(0, 5).map((keyword, idx) => (
                          <span key={idx} className="keyword-tag">
                            {keyword}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="cluster-severity">
                      <div
                        className="severity-bar"
                        style={{
                          width: `${cluster.severity * 100}%`,
                          backgroundColor: getSeverityColor(cluster.severity),
                        }}
                      />
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Slices Tab */}
          {activeTab === 'slices' && (
            <div className="slices-list">
              {slices.length === 0 ? (
                <div className="empty-state">No problematic slices detected</div>
              ) : (
                slices.map((slice) => (
                  <div
                    key={slice.id}
                    className="slice-card"
                    onClick={() => onSliceClick?.(slice)}
                  >
                    <div className="slice-header">
                      <span className="slice-name">{slice.name}</span>
                      {slice.isSignificant && (
                        <span className="significant-badge">Significant</span>
                      )}
                    </div>
                    <div className="slice-metrics">
                      <div className="metric">
                        <label>{slice.metric}</label>
                        <value>{slice.metricValue.toFixed(2)}</value>
                      </div>
                      <div className="metric">
                        <label>Baseline</label>
                        <value>{slice.baseline.toFixed(2)}</value>
                      </div>
                      <div className="metric deviation">
                        <label>Deviation</label>
                        <value
                          style={{
                            color: slice.deviationPercentage > 0
                              ? 'var(--color-danger)'
                              : 'var(--color-success)',
                          }}
                        >
                          {slice.deviationPercentage > 0 ? '+' : ''}
                          {slice.deviationPercentage.toFixed(1)}%
                        </value>
                      </div>
                    </div>
                    <div className="slice-details">
                      <span>
                        {slice.dimension}: {slice.dimensionValue}
                      </span>
                      <span>Impact: {(slice.impactScore * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default RCAExplorer;
