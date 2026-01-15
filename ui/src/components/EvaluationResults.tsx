/**
 * EvaluationResults Component
 * Display evaluation results with filtering and visualization
 */

import React, { useState, useEffect } from 'react';
import { evaluationsApi, Evaluation, EvaluationSummary } from '../services/api';

interface EvaluationResultsProps {
  traceId?: string;
  modelId?: string;
  onEvaluationClick?: (evaluation: Evaluation) => void;
}

export function EvaluationResults({
  traceId,
  modelId,
  onEvaluationClick,
}: EvaluationResultsProps) {
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<{
    evaluatorType?: string;
    verdict?: string;
  }>({});

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [evalRes, summaryRes] = await Promise.all([
          traceId
            ? evaluationsApi.getByTrace(traceId)
            : evaluationsApi.list(filter),
          evaluationsApi.getSummary(),
        ]);

        setEvaluations(
          traceId
            ? (evalRes as { traceId: string; evaluations: Evaluation[] }).evaluations
            : (evalRes as { data: Evaluation[] }).data
        );
        setSummary(summaryRes);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load evaluations');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [traceId, JSON.stringify(filter)]);

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'pass':
        return 'var(--color-success, #22c55e)';
      case 'fail':
        return 'var(--color-danger, #ef4444)';
      case 'warn':
        return 'var(--color-warning, #eab308)';
      default:
        return 'var(--color-muted, #6b7280)';
    }
  };

  const getEvaluatorIcon = (type: string) => {
    switch (type) {
      case 'hallucination':
        return '🔍';
      case 'toxicity':
        return '⚠️';
      case 'rag_quality':
        return '📚';
      default:
        return '📊';
    }
  };

  return (
    <div className="evaluation-results">
      {/* Summary Cards */}
      {summary && (
        <div className="eval-summary">
          <div className="summary-card total">
            <label>Total Evaluations</label>
            <value>{summary.totalEvaluations.toLocaleString()}</value>
          </div>
          <div className="summary-card pass">
            <label>Pass Rate</label>
            <value style={{ color: getVerdictColor('pass') }}>
              {(summary.passRate * 100).toFixed(1)}%
            </value>
          </div>
          <div className="summary-card fail">
            <label>Fail Rate</label>
            <value style={{ color: getVerdictColor('fail') }}>
              {(summary.failRate * 100).toFixed(1)}%
            </value>
          </div>
          <div className="summary-card warn">
            <label>Warn Rate</label>
            <value style={{ color: getVerdictColor('warn') }}>
              {(summary.warnRate * 100).toFixed(1)}%
            </value>
          </div>
        </div>
      )}

      {/* By Type Breakdown */}
      {summary && summary.byType && (
        <div className="eval-by-type">
          <h3>By Evaluator Type</h3>
          <div className="type-breakdown">
            {Object.entries(summary.byType).map(([type, data]) => (
              <div key={type} className="type-item">
                <div className="type-header">
                  <span className="type-icon">{getEvaluatorIcon(type)}</span>
                  <span className="type-name">{type}</span>
                </div>
                <div className="type-stats">
                  <span className="type-total">{data.total} evaluations</span>
                  <span
                    className="type-pass-rate"
                    style={{ color: getVerdictColor('pass') }}
                  >
                    {(data.passRate * 100).toFixed(0)}% pass
                  </span>
                </div>
                <div className="type-bar">
                  <div
                    className="pass-portion"
                    style={{ width: `${data.passRate * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      {!traceId && (
        <div className="eval-filters">
          <select
            value={filter.evaluatorType || ''}
            onChange={(e) =>
              setFilter({ ...filter, evaluatorType: e.target.value || undefined })
            }
          >
            <option value="">All Types</option>
            <option value="hallucination">Hallucination</option>
            <option value="toxicity">Toxicity</option>
            <option value="rag_quality">RAG Quality</option>
          </select>
          <select
            value={filter.verdict || ''}
            onChange={(e) =>
              setFilter({ ...filter, verdict: e.target.value || undefined })
            }
          >
            <option value="">All Verdicts</option>
            <option value="pass">Pass</option>
            <option value="fail">Fail</option>
            <option value="warn">Warn</option>
          </select>
        </div>
      )}

      {/* Evaluation List */}
      <div className="eval-list">
        <h3>
          {traceId ? 'Trace Evaluations' : 'Recent Evaluations'}
        </h3>

        {loading && <div className="eval-loading">Loading...</div>}
        {error && <div className="eval-error">{error}</div>}

        {!loading && !error && (
          <>
            {evaluations.length === 0 ? (
              <div className="eval-empty">No evaluations found</div>
            ) : (
              <div className="eval-items">
                {evaluations.map((evaluation) => (
                  <div
                    key={evaluation.id}
                    className="eval-item"
                    onClick={() => onEvaluationClick?.(evaluation)}
                  >
                    <div className="eval-item-header">
                      <span className="eval-type">
                        {getEvaluatorIcon(evaluation.evaluatorType)}{' '}
                        {evaluation.evaluatorType}
                      </span>
                      <span
                        className="eval-verdict"
                        style={{
                          backgroundColor: getVerdictColor(evaluation.verdict),
                        }}
                      >
                        {evaluation.verdict.toUpperCase()}
                      </span>
                    </div>

                    <div className="eval-item-score">
                      <div className="score-bar-container">
                        <div
                          className="score-bar"
                          style={{ width: `${evaluation.score * 100}%` }}
                        />
                      </div>
                      <span className="score-value">
                        {(evaluation.score * 100).toFixed(0)}%
                      </span>
                    </div>

                    {evaluation.explanation && (
                      <div className="eval-item-explanation">
                        {evaluation.explanation.length > 150
                          ? evaluation.explanation.substring(0, 150) + '...'
                          : evaluation.explanation}
                      </div>
                    )}

                    <div className="eval-item-footer">
                      <span className="eval-trace">
                        Trace: {evaluation.traceId.substring(0, 8)}...
                      </span>
                      <span className="eval-time">
                        {new Date(evaluation.timestamp).toLocaleString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default EvaluationResults;
