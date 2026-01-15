import React from 'react';
import { InlineField, Select, Input, MultiSelect } from '@grafana/ui';
import { QueryEditorProps, SelectableValue } from '@grafana/data';
import { PyFlareDataSource } from './datasource';
import {
  PyFlareDataSourceOptions,
  PyFlareQuery,
  QueryType,
  AVAILABLE_METRICS,
  AGGREGATION_OPTIONS,
  GROUP_BY_OPTIONS,
  DRIFT_TYPE_OPTIONS,
} from './types';

type Props = QueryEditorProps<PyFlareDataSource, PyFlareQuery, PyFlareDataSourceOptions>;

const QUERY_TYPE_OPTIONS: Array<SelectableValue<QueryType>> = [
  { value: 'metrics', label: 'Metrics', description: 'Query time series metrics' },
  { value: 'traces', label: 'Traces', description: 'Query trace data' },
  { value: 'drift', label: 'Drift', description: 'Query drift detection data' },
  { value: 'costs', label: 'Costs', description: 'Query cost analytics' },
  { value: 'evaluations', label: 'Evaluations', description: 'Query evaluation scores' },
];

const STATUS_OPTIONS: Array<SelectableValue<string>> = [
  { value: 'all', label: 'All' },
  { value: 'ok', label: 'Success' },
  { value: 'error', label: 'Error' },
];

const COST_GROUP_BY_OPTIONS: Array<SelectableValue<string>> = [
  { value: 'model', label: 'Model' },
  { value: 'service', label: 'Service' },
  { value: 'user', label: 'User' },
  { value: 'feature', label: 'Feature' },
];

const EVAL_METRIC_OPTIONS: Array<SelectableValue<string>> = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'latency', label: 'Latency' },
  { value: 'toxicity', label: 'Toxicity' },
  { value: 'hallucination', label: 'Hallucination' },
  { value: 'relevance', label: 'Relevance' },
];

/**
 * Query editor for PyFlare data source
 */
export function QueryEditor({ query, onChange, onRunQuery }: Props) {
  const onQueryTypeChange = (value: SelectableValue<QueryType>) => {
    onChange({ ...query, queryType: value.value! });
    onRunQuery();
  };

  const onMetricChange = (value: SelectableValue<string>) => {
    onChange({ ...query, metric: value.value });
    onRunQuery();
  };

  const onAggregationChange = (value: SelectableValue<string>) => {
    onChange({ ...query, aggregation: value.value as any });
    onRunQuery();
  };

  const onGroupByChange = (values: Array<SelectableValue<string>>) => {
    onChange({ ...query, groupBy: values.map((v) => v.value!) });
    onRunQuery();
  };

  const onServiceNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...query, serviceName: e.target.value });
  };

  const onModelIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...query, modelId: e.target.value });
  };

  const onStatusChange = (value: SelectableValue<string>) => {
    onChange({ ...query, status: value.value as any });
    onRunQuery();
  };

  const onMinDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...query, minDuration: parseInt(e.target.value, 10) || undefined });
  };

  const onMaxDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...query, maxDuration: parseInt(e.target.value, 10) || undefined });
  };

  const onDriftTypeChange = (value: SelectableValue<string>) => {
    onChange({ ...query, driftType: value.value as any });
    onRunQuery();
  };

  const onThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...query, threshold: parseFloat(e.target.value) || undefined });
  };

  const onCostGroupByChange = (value: SelectableValue<string>) => {
    onChange({ ...query, costGroupBy: value.value as any });
    onRunQuery();
  };

  const onEvalMetricChange = (value: SelectableValue<string>) => {
    onChange({ ...query, evalMetric: value.value as any });
    onRunQuery();
  };

  const onLimitChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...query, limit: parseInt(e.target.value, 10) || 100 });
  };

  return (
    <>
      {/* Query Type Selection */}
      <InlineField label="Query Type" labelWidth={14}>
        <Select
          width={20}
          options={QUERY_TYPE_OPTIONS}
          value={query.queryType}
          onChange={onQueryTypeChange}
        />
      </InlineField>

      {/* Metrics Query Options */}
      {query.queryType === 'metrics' && (
        <>
          <InlineField label="Metric" labelWidth={14}>
            <Select
              width={25}
              options={AVAILABLE_METRICS}
              value={query.metric}
              onChange={onMetricChange}
            />
          </InlineField>
          <InlineField label="Aggregation" labelWidth={14}>
            <Select
              width={15}
              options={AGGREGATION_OPTIONS}
              value={query.aggregation}
              onChange={onAggregationChange}
            />
          </InlineField>
          <InlineField label="Group By" labelWidth={14}>
            <MultiSelect
              width={30}
              options={GROUP_BY_OPTIONS}
              value={query.groupBy?.map((v) => ({ value: v, label: v }))}
              onChange={onGroupByChange}
              placeholder="Select fields..."
            />
          </InlineField>
        </>
      )}

      {/* Traces Query Options */}
      {query.queryType === 'traces' && (
        <>
          <InlineField label="Status" labelWidth={14}>
            <Select
              width={15}
              options={STATUS_OPTIONS}
              value={query.status || 'all'}
              onChange={onStatusChange}
            />
          </InlineField>
          <InlineField label="Min Duration (ms)" labelWidth={14}>
            <Input
              width={15}
              type="number"
              value={query.minDuration || ''}
              onChange={onMinDurationChange}
              onBlur={onRunQuery}
              placeholder="0"
            />
          </InlineField>
          <InlineField label="Max Duration (ms)" labelWidth={14}>
            <Input
              width={15}
              type="number"
              value={query.maxDuration || ''}
              onChange={onMaxDurationChange}
              onBlur={onRunQuery}
              placeholder="10000"
            />
          </InlineField>
        </>
      )}

      {/* Drift Query Options */}
      {query.queryType === 'drift' && (
        <>
          <InlineField label="Drift Type" labelWidth={14}>
            <Select
              width={20}
              options={DRIFT_TYPE_OPTIONS}
              value={query.driftType || 'all'}
              onChange={onDriftTypeChange}
            />
          </InlineField>
          <InlineField label="Threshold" labelWidth={14}>
            <Input
              width={15}
              type="number"
              step="0.1"
              value={query.threshold || ''}
              onChange={onThresholdChange}
              onBlur={onRunQuery}
              placeholder="0.3"
            />
          </InlineField>
        </>
      )}

      {/* Costs Query Options */}
      {query.queryType === 'costs' && (
        <InlineField label="Group By" labelWidth={14}>
          <Select
            width={20}
            options={COST_GROUP_BY_OPTIONS}
            value={query.costGroupBy}
            onChange={onCostGroupByChange}
            isClearable
          />
        </InlineField>
      )}

      {/* Evaluations Query Options */}
      {query.queryType === 'evaluations' && (
        <InlineField label="Metric" labelWidth={14}>
          <Select
            width={20}
            options={EVAL_METRIC_OPTIONS}
            value={query.evalMetric}
            onChange={onEvalMetricChange}
          />
        </InlineField>
      )}

      {/* Common Filters */}
      <div style={{ marginTop: '8px' }}>
        <InlineField label="Service" labelWidth={14}>
          <Input
            width={20}
            value={query.serviceName || ''}
            onChange={onServiceNameChange}
            onBlur={onRunQuery}
            placeholder="Service name"
          />
        </InlineField>
        <InlineField label="Model" labelWidth={14}>
          <Input
            width={20}
            value={query.modelId || ''}
            onChange={onModelIdChange}
            onBlur={onRunQuery}
            placeholder="Model ID"
          />
        </InlineField>
        <InlineField label="Limit" labelWidth={14}>
          <Input
            width={10}
            type="number"
            value={query.limit || 100}
            onChange={onLimitChange}
            onBlur={onRunQuery}
          />
        </InlineField>
      </div>
    </>
  );
}
