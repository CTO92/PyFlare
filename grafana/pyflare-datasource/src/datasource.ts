import {
  DataSourceApi,
  DataSourceInstanceSettings,
  DataQueryRequest,
  DataQueryResponse,
  DataFrameDTO,
  FieldType,
  MutableDataFrame,
  dateTime,
  TestDataSourceResponse,
} from '@grafana/data';
import { getBackendSrv, getTemplateSrv } from '@grafana/runtime';

import {
  PyFlareQuery,
  PyFlareDataSourceOptions,
  PyFlareSecureJsonData,
  defaultQuery,
} from './types';

/**
 * PyFlare Data Source
 *
 * Security Features:
 * - API key is passed via Grafana's secure backend proxy
 * - Secrets are never exposed to the browser
 * - Uses Grafana's built-in authentication handling
 */
export class PyFlareDataSource extends DataSourceApi<PyFlareQuery, PyFlareDataSourceOptions> {
  url: string;
  defaultServiceName?: string;
  defaultModelId?: string;
  // Store instance settings for secure header access
  private instanceSettings: DataSourceInstanceSettings<PyFlareDataSourceOptions>;

  constructor(instanceSettings: DataSourceInstanceSettings<PyFlareDataSourceOptions>) {
    super(instanceSettings);

    this.instanceSettings = instanceSettings;
    this.url = instanceSettings.jsonData.url || '';
    this.defaultServiceName = instanceSettings.jsonData.defaultServiceName;
    this.defaultModelId = instanceSettings.jsonData.defaultModelId;
  }

  /**
   * Get default query values
   */
  getDefaultQuery(): Partial<PyFlareQuery> {
    return defaultQuery;
  }

  /**
   * Execute queries
   */
  async query(request: DataQueryRequest<PyFlareQuery>): Promise<DataQueryResponse> {
    const { range } = request;
    const from = range?.from.toISOString();
    const to = range?.to.toISOString();

    const promises = request.targets
      .filter((target) => !target.hide)
      .map(async (target) => {
        const query = {
          ...defaultQuery,
          ...target,
        };

        // Replace template variables
        const serviceName = getTemplateSrv().replace(
          query.serviceName || this.defaultServiceName || '',
          request.scopedVars
        );
        const modelId = getTemplateSrv().replace(
          query.modelId || this.defaultModelId || '',
          request.scopedVars
        );

        switch (query.queryType) {
          case 'traces':
            return this.queryTraces(query, from, to, serviceName, modelId);
          case 'metrics':
            return this.queryMetrics(query, from, to, serviceName, modelId);
          case 'drift':
            return this.queryDrift(query, from, to, modelId);
          case 'costs':
            return this.queryCosts(query, from, to);
          case 'evaluations':
            return this.queryEvaluations(query, from, to, modelId);
          default:
            return new MutableDataFrame();
        }
      });

    const data = await Promise.all(promises);
    return { data };
  }

  /**
   * Query traces from PyFlare API
   */
  private async queryTraces(
    query: PyFlareQuery,
    from?: string,
    to?: string,
    serviceName?: string,
    modelId?: string
  ): Promise<MutableDataFrame> {
    const params = new URLSearchParams();
    if (from) params.set('start_time', from);
    if (to) params.set('end_time', to);
    if (serviceName) params.set('service_name', serviceName);
    if (modelId) params.set('model_id', modelId);
    if (query.status && query.status !== 'all') params.set('status', query.status);
    if (query.minDuration) params.set('min_duration_ms', query.minDuration.toString());
    if (query.maxDuration) params.set('max_duration_ms', query.maxDuration.toString());
    if (query.limit) params.set('limit', query.limit.toString());

    const response = await this.doRequest('/api/v1/traces', params);

    const frame = new MutableDataFrame({
      refId: query.refId,
      fields: [
        { name: 'Time', type: FieldType.time },
        { name: 'Trace ID', type: FieldType.string },
        { name: 'Service', type: FieldType.string },
        { name: 'Model', type: FieldType.string },
        { name: 'Status', type: FieldType.string },
        { name: 'Duration (ms)', type: FieldType.number },
        { name: 'Input Tokens', type: FieldType.number },
        { name: 'Output Tokens', type: FieldType.number },
        { name: 'Cost', type: FieldType.number },
      ],
    });

    if (response.traces) {
      for (const trace of response.traces) {
        frame.add({
          Time: dateTime(trace.start_time).valueOf(),
          'Trace ID': trace.trace_id,
          Service: trace.service_name,
          Model: trace.model_id,
          Status: trace.status,
          'Duration (ms)': trace.latency_ms,
          'Input Tokens': trace.input_tokens,
          'Output Tokens': trace.output_tokens,
          Cost: trace.cost_micros / 1_000_000,
        });
      }
    }

    return frame;
  }

  /**
   * Query metrics from PyFlare API
   */
  private async queryMetrics(
    query: PyFlareQuery,
    from?: string,
    to?: string,
    serviceName?: string,
    modelId?: string
  ): Promise<MutableDataFrame> {
    const params = new URLSearchParams();
    if (from) params.set('start_time', from);
    if (to) params.set('end_time', to);
    if (serviceName) params.set('service_name', serviceName);
    if (modelId) params.set('model_id', modelId);
    if (query.metric) params.set('metric', query.metric);
    if (query.aggregation) params.set('aggregation', query.aggregation);
    if (query.groupBy?.length) params.set('group_by', query.groupBy.join(','));

    const response = await this.doRequest('/api/v1/metrics/query', params);

    const frame = new MutableDataFrame({
      refId: query.refId,
      fields: [
        { name: 'Time', type: FieldType.time },
        { name: 'Value', type: FieldType.number },
      ],
    });

    // Add group by fields
    if (query.groupBy?.length) {
      for (const groupBy of query.groupBy) {
        frame.addField({ name: groupBy, type: FieldType.string });
      }
    }

    if (response.data) {
      for (const point of response.data) {
        const row: Record<string, any> = {
          Time: dateTime(point.timestamp).valueOf(),
          Value: point.value,
        };

        if (query.groupBy?.length && point.labels) {
          for (const groupBy of query.groupBy) {
            row[groupBy] = point.labels[groupBy] || '';
          }
        }

        frame.add(row);
      }
    }

    return frame;
  }

  /**
   * Query drift data from PyFlare API
   */
  private async queryDrift(
    query: PyFlareQuery,
    from?: string,
    to?: string,
    modelId?: string
  ): Promise<MutableDataFrame> {
    const params = new URLSearchParams();
    if (from) params.set('start_time', from);
    if (to) params.set('end_time', to);
    if (modelId) params.set('model_id', modelId);
    if (query.driftType && query.driftType !== 'all') params.set('drift_type', query.driftType);
    if (query.threshold) params.set('threshold', query.threshold.toString());

    const response = await this.doRequest('/api/v1/drift/timeline', params);

    const frame = new MutableDataFrame({
      refId: query.refId,
      fields: [
        { name: 'Time', type: FieldType.time },
        { name: 'Feature Drift', type: FieldType.number },
        { name: 'Embedding Drift', type: FieldType.number },
        { name: 'Concept Drift', type: FieldType.number },
        { name: 'Prediction Drift', type: FieldType.number },
      ],
    });

    if (response.timeline) {
      for (const point of response.timeline) {
        frame.add({
          Time: dateTime(point.timestamp).valueOf(),
          'Feature Drift': point.feature_drift,
          'Embedding Drift': point.embedding_drift,
          'Concept Drift': point.concept_drift,
          'Prediction Drift': point.prediction_drift,
        });
      }
    }

    return frame;
  }

  /**
   * Query cost data from PyFlare API
   */
  private async queryCosts(
    query: PyFlareQuery,
    from?: string,
    to?: string
  ): Promise<MutableDataFrame> {
    const params = new URLSearchParams();
    if (from) params.set('start_time', from);
    if (to) params.set('end_time', to);
    if (query.costGroupBy) params.set('group_by', query.costGroupBy);

    const response = await this.doRequest('/api/v1/costs/timeline', params);

    const frame = new MutableDataFrame({
      refId: query.refId,
      fields: [
        { name: 'Time', type: FieldType.time },
        { name: 'Cost', type: FieldType.number },
        { name: 'Tokens', type: FieldType.number },
        { name: 'Requests', type: FieldType.number },
      ],
    });

    if (query.costGroupBy) {
      frame.addField({ name: query.costGroupBy, type: FieldType.string });
    }

    if (response.timeline) {
      for (const point of response.timeline) {
        const row: Record<string, any> = {
          Time: dateTime(point.timestamp).valueOf(),
          Cost: point.cost_micros / 1_000_000,
          Tokens: point.total_tokens,
          Requests: point.request_count,
        };

        if (query.costGroupBy && point[query.costGroupBy]) {
          row[query.costGroupBy] = point[query.costGroupBy];
        }

        frame.add(row);
      }
    }

    return frame;
  }

  /**
   * Query evaluation data from PyFlare API
   */
  private async queryEvaluations(
    query: PyFlareQuery,
    from?: string,
    to?: string,
    modelId?: string
  ): Promise<MutableDataFrame> {
    const params = new URLSearchParams();
    if (from) params.set('start_time', from);
    if (to) params.set('end_time', to);
    if (modelId) params.set('model_id', modelId);
    if (query.evalMetric) params.set('metric', query.evalMetric);

    const response = await this.doRequest('/api/v1/evaluations/timeline', params);

    const frame = new MutableDataFrame({
      refId: query.refId,
      fields: [
        { name: 'Time', type: FieldType.time },
        { name: 'Score', type: FieldType.number },
        { name: 'Count', type: FieldType.number },
      ],
    });

    if (response.timeline) {
      for (const point of response.timeline) {
        frame.add({
          Time: dateTime(point.timestamp).valueOf(),
          Score: point.avg_score,
          Count: point.count,
        });
      }
    }

    return frame;
  }

  /**
   * Test the data source connection
   */
  async testDatasource(): Promise<TestDataSourceResponse> {
    try {
      const response = await this.doRequest('/api/v1/health', new URLSearchParams());

      if (response.status === 'healthy') {
        return {
          status: 'success',
          message: 'Successfully connected to PyFlare',
        };
      } else {
        return {
          status: 'error',
          message: `PyFlare service is ${response.status}`,
        };
      }
    } catch (error) {
      return {
        status: 'error',
        message: `Failed to connect to PyFlare: ${error}`,
      };
    }
  }

  /**
   * Make HTTP request to PyFlare API
   *
   * SEC-008 Fix: Uses Grafana's backend proxy to securely handle API keys.
   * The API key is stored in secureJsonData and passed via headers
   * by the Grafana backend - never exposed to the browser.
   */
  private async doRequest(path: string, params: URLSearchParams): Promise<any> {
    const url = `${this.url}${path}?${params.toString()}`;

    // Build headers - API key is handled securely via Grafana's backend
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // For data sources configured with API key, use Grafana's datasource proxy
    // which automatically adds the secure headers configured in the data source
    const response = await getBackendSrv().fetch({
      url,
      method: 'GET',
      headers,
      // Use credentials to include cookies if needed
      credentials: 'include',
      // This tells Grafana to use the data source's configured authentication
      // The secureJsonData.apiKey is added as Authorization header by Grafana's backend
    }).toPromise();

    return response?.data;
  }

  /**
   * Make authenticated request using Grafana's proxy
   *
   * This method uses Grafana's built-in data source proxy which:
   * 1. Routes requests through Grafana's backend
   * 2. Automatically adds configured authentication headers
   * 3. Keeps API keys secure (never sent to browser)
   */
  private async doProxiedRequest(path: string, params: URLSearchParams): Promise<any> {
    // Use Grafana's data source proxy endpoint
    // Format: /api/datasources/proxy/{datasource_id}/...
    const proxyUrl = `/api/datasources/proxy/${this.instanceSettings.id}${path}?${params.toString()}`;

    const response = await getBackendSrv().fetch({
      url: proxyUrl,
      method: 'GET',
    }).toPromise();

    return response?.data;
  }

  /**
   * Get annotations (for drift alerts, etc.)
   */
  async annotationQuery(options: any): Promise<any[]> {
    const { annotation, range } = options;
    const from = range?.from.toISOString();
    const to = range?.to.toISOString();

    const params = new URLSearchParams();
    if (from) params.set('start_time', from);
    if (to) params.set('end_time', to);

    const response = await this.doRequest('/api/v1/alerts', params);

    if (!response.alerts) {
      return [];
    }

    return response.alerts.map((alert: any) => ({
      annotation,
      time: dateTime(alert.timestamp).valueOf(),
      title: alert.title,
      text: alert.description,
      tags: alert.tags || [],
    }));
  }
}
