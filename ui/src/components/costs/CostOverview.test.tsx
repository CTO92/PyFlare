import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import CostOverview, { type CostMetrics, type BudgetStatus } from './CostOverview';

const mockMetrics: CostMetrics = {
  totalCost: 12450.75,
  costChange: 12.5,
  totalTokens: 45230000,
  tokenChange: 8.2,
  avgCostPerRequest: 0.0245,
  avgCostChange: -3.1,
  requestCount: 508600,
  requestChange: 15.7,
};

const mockBudget: BudgetStatus = {
  used: 9800,
  limit: 15000,
  period: 'monthly',
  alertThreshold: 80,
};

describe('CostOverview', () => {
  it('renders total cost', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 30 Days"
      />
    );

    expect(screen.getByText('Total Cost')).toBeInTheDocument();
    expect(screen.getByText('$12,450.75')).toBeInTheDocument();
  });

  it('renders token count with formatting', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 30 Days"
      />
    );

    expect(screen.getByText('Total Tokens')).toBeInTheDocument();
    expect(screen.getByText('45.2M')).toBeInTheDocument();
  });

  it('renders request count', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 30 Days"
      />
    );

    expect(screen.getByText('Total Requests')).toBeInTheDocument();
    expect(screen.getByText('508.6K')).toBeInTheDocument();
  });

  it('displays cost change percentage', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 30 Days"
      />
    );

    expect(screen.getByText('12.5%')).toBeInTheDocument();
  });

  it('displays budget progress', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 30 Days"
      />
    );

    expect(screen.getByText('Monthly Budget')).toBeInTheDocument();
    expect(screen.getByText(/\$9,800/)).toBeInTheDocument();
    expect(screen.getByText(/\$15,000/)).toBeInTheDocument();
  });

  it('shows budget percentage', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 30 Days"
      />
    );

    // 9800 / 15000 = 65.3%
    expect(screen.getByText('65.3%')).toBeInTheDocument();
  });

  it('displays time range', () => {
    render(
      <CostOverview
        metrics={mockMetrics}
        budget={mockBudget}
        timeRange="Last 7 Days"
      />
    );

    expect(screen.getByText(/Last 7 Days/)).toBeInTheDocument();
  });

  it('shows alert when budget is near limit', () => {
    const nearLimitBudget: BudgetStatus = {
      used: 13000,
      limit: 15000,
      period: 'monthly',
      alertThreshold: 80,
    };

    render(
      <CostOverview
        metrics={mockMetrics}
        budget={nearLimitBudget}
        timeRange="Last 30 Days"
      />
    );

    // Budget is at 86.7%, which exceeds 80% threshold
    const progressElement = screen.getByText('86.7%');
    expect(progressElement).toBeInTheDocument();
  });

  it('shows alert when budget is exceeded', () => {
    const exceededBudget: BudgetStatus = {
      used: 16000,
      limit: 15000,
      period: 'monthly',
      alertThreshold: 80,
    };

    render(
      <CostOverview
        metrics={mockMetrics}
        budget={exceededBudget}
        timeRange="Last 30 Days"
      />
    );

    expect(screen.getByText('106.7%')).toBeInTheDocument();
  });
});
