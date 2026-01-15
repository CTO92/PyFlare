import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import DriftOverview, { type DriftScores } from './DriftOverview';

const mockScores: DriftScores = {
  feature: 0.25,
  embedding: 0.35,
  concept: 0.15,
  prediction: 0.2,
};

const mockThresholds: DriftScores = {
  feature: 0.3,
  embedding: 0.3,
  concept: 0.4,
  prediction: 0.3,
};

describe('DriftOverview', () => {
  it('renders model ID', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={mockScores}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={0}
      />
    );

    expect(screen.getByText(/Model: gpt-4/)).toBeInTheDocument();
  });

  it('renders all drift score cards', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={mockScores}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={0}
      />
    );

    expect(screen.getByText('Feature Drift')).toBeInTheDocument();
    expect(screen.getByText('Embedding Drift')).toBeInTheDocument();
    expect(screen.getByText('Concept Drift')).toBeInTheDocument();
    expect(screen.getByText('Prediction Drift')).toBeInTheDocument();
  });

  it('shows healthy status when scores are below thresholds', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={{ feature: 0.1, embedding: 0.1, concept: 0.1, prediction: 0.1 }}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={0}
      />
    );

    expect(screen.getByText('Healthy')).toBeInTheDocument();
  });

  it('shows warning status when scores approach thresholds', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={{ feature: 0.25, embedding: 0.25, concept: 0.25, prediction: 0.25 }}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={0}
      />
    );

    expect(screen.getByText('Warning')).toBeInTheDocument();
  });

  it('shows critical status when scores exceed thresholds', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={{ feature: 0.5, embedding: 0.5, concept: 0.5, prediction: 0.5 }}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={0}
      />
    );

    expect(screen.getByText('Critical')).toBeInTheDocument();
  });

  it('displays active alerts count', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={mockScores}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={3}
      />
    );

    expect(screen.getByText('3 active alerts')).toBeInTheDocument();
  });

  it('formats drift scores as percentages', () => {
    render(
      <DriftOverview
        modelId="gpt-4"
        scores={{ feature: 0.25, embedding: 0.35, concept: 0.15, prediction: 0.2 }}
        thresholds={mockThresholds}
        lastUpdated={new Date().toISOString()}
        activeAlerts={0}
      />
    );

    expect(screen.getByText('25.0%')).toBeInTheDocument();
    expect(screen.getByText('35.0%')).toBeInTheDocument();
    expect(screen.getByText('15.0%')).toBeInTheDocument();
    expect(screen.getByText('20.0%')).toBeInTheDocument();
  });
});
