/**
 * CostCharts Component
 * Cost breakdown visualizations including timeline, pie chart, and budget status
 */

import React, { useState } from 'react';
import { useCostSummary, useCostBreakdown, useCostTimeline, useBudgets, formatCost, formatTokens } from '../hooks/useCosts';

interface CostChartsProps {
  timeRange?: { start?: string; end?: string };
}

type Dimension = 'model' | 'user' | 'feature';

export function CostCharts({ timeRange = {} }: CostChartsProps) {
  const [dimension, setDimension] = useState<Dimension>('model');
  const { summary, loading: summaryLoading } = useCostSummary(timeRange);
  const { breakdown, loading: breakdownLoading } = useCostBreakdown({ ...timeRange, dimension });
  const { timeline, loading: timelineLoading } = useCostTimeline({ ...timeRange, granularity: 'hour' });
  const { budgets, loading: budgetsLoading } = useBudgets();

  // Calculate max values for scaling
  const maxTimelineCost = Math.max(...timeline.map(t => t.costMicros), 1);

  return (
    <div className="cost-charts">
      {/* Summary Cards */}
      <div className="cost-summary-cards">
        <div className="cost-card">
          <label>Total Spend</label>
          <value className="large">
            {summaryLoading ? '...' : formatCost(summary?.totalCostMicros ?? 0)}
          </value>
        </div>
        <div className="cost-card">
          <label>Total Tokens</label>
          <value className="large">
            {summaryLoading ? '...' : formatTokens(summary?.totalTokens ?? 0)}
          </value>
        </div>
        <div className="cost-card">
          <label>Requests</label>
          <value className="large">
            {summaryLoading ? '...' : (summary?.requestCount ?? 0).toLocaleString()}
          </value>
        </div>
        <div className="cost-card">
          <label>Avg Cost/Request</label>
          <value className="large">
            {summaryLoading ? '...' : formatCost(summary?.avgCostPerRequestMicros ?? 0)}
          </value>
        </div>
      </div>

      {/* Cost Timeline */}
      <div className="cost-section">
        <h3>Cost Over Time</h3>
        <div className="cost-timeline-chart">
          {timelineLoading ? (
            <div className="chart-loading">Loading...</div>
          ) : timeline.length === 0 ? (
            <div className="chart-empty">No data</div>
          ) : (
            <div className="bar-chart">
              {timeline.map((point, idx) => (
                <div
                  key={idx}
                  className="bar-item"
                  title={`${new Date(point.timestamp).toLocaleString()}: ${formatCost(point.costMicros)}`}
                >
                  <div
                    className="bar"
                    style={{ height: `${(point.costMicros / maxTimelineCost) * 100}%` }}
                  />
                  <div className="bar-label">
                    {new Date(point.timestamp).getHours()}:00
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Cost Breakdown */}
      <div className="cost-section">
        <div className="section-header">
          <h3>Cost Breakdown</h3>
          <div className="dimension-selector">
            <button
              className={dimension === 'model' ? 'active' : ''}
              onClick={() => setDimension('model')}
            >
              By Model
            </button>
            <button
              className={dimension === 'user' ? 'active' : ''}
              onClick={() => setDimension('user')}
            >
              By User
            </button>
            <button
              className={dimension === 'feature' ? 'active' : ''}
              onClick={() => setDimension('feature')}
            >
              By Feature
            </button>
          </div>
        </div>

        <div className="breakdown-chart">
          {breakdownLoading ? (
            <div className="chart-loading">Loading...</div>
          ) : !breakdown || breakdown.items.length === 0 ? (
            <div className="chart-empty">No data</div>
          ) : (
            <div className="breakdown-list">
              {breakdown.items.map((item, idx) => (
                <div key={idx} className="breakdown-item">
                  <div className="breakdown-info">
                    <span className="breakdown-value">{item.value}</span>
                    <span className="breakdown-cost">{formatCost(item.costMicros)}</span>
                  </div>
                  <div className="breakdown-bar-container">
                    <div
                      className="breakdown-bar"
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                  <div className="breakdown-details">
                    <span>{formatTokens(item.tokens)} tokens</span>
                    <span>{item.requests.toLocaleString()} requests</span>
                    <span>{item.percentage.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Budget Status */}
      <div className="cost-section">
        <h3>Budget Status</h3>
        <div className="budget-list">
          {budgetsLoading ? (
            <div className="chart-loading">Loading...</div>
          ) : budgets.length === 0 ? (
            <div className="chart-empty">No budgets configured</div>
          ) : (
            budgets.map((budget) => (
              <div
                key={budget.id}
                className={`budget-item ${budget.limitExceeded ? 'exceeded' : budget.warningTriggered ? 'warning' : ''}`}
              >
                <div className="budget-header">
                  <span className="budget-name">
                    {budget.dimension}: {budget.dimensionValue}
                  </span>
                  <span className="budget-period">{budget.period}</span>
                </div>
                <div className="budget-progress">
                  <div className="budget-bar-container">
                    <div
                      className="budget-bar"
                      style={{ width: `${Math.min(budget.utilizationPercentage, 100)}%` }}
                    />
                    {budget.utilizationPercentage > 100 && (
                      <div
                        className="budget-bar-overflow"
                        style={{ width: `${budget.utilizationPercentage - 100}%` }}
                      />
                    )}
                  </div>
                </div>
                <div className="budget-details">
                  <span className="budget-current">
                    {formatCost(budget.currentSpendMicros)} spent
                  </span>
                  <span className="budget-limit">
                    / {formatCost(budget.hardLimitMicros)} limit
                  </span>
                  <span className="budget-percentage">
                    {budget.utilizationPercentage.toFixed(1)}%
                  </span>
                </div>
                {budget.warningTriggered && !budget.limitExceeded && (
                  <div className="budget-warning">
                    Warning: Approaching budget limit
                  </div>
                )}
                {budget.limitExceeded && (
                  <div className="budget-exceeded">
                    Budget exceeded!
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default CostCharts;
