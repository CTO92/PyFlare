import React, { ChangeEvent } from 'react';
import { InlineField, Input, SecretInput } from '@grafana/ui';
import { DataSourcePluginOptionsEditorProps } from '@grafana/data';
import { PyFlareDataSourceOptions, PyFlareSecureJsonData } from './types';

interface Props extends DataSourcePluginOptionsEditorProps<PyFlareDataSourceOptions, PyFlareSecureJsonData> {}

/**
 * Configuration editor for PyFlare data source
 */
export function ConfigEditor(props: Props) {
  const { onOptionsChange, options } = props;
  const { jsonData, secureJsonFields, secureJsonData } = options;

  const onURLChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: {
        ...jsonData,
        url: event.target.value,
      },
    });
  };

  const onDefaultServiceNameChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: {
        ...jsonData,
        defaultServiceName: event.target.value,
      },
    });
  };

  const onDefaultModelIdChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: {
        ...jsonData,
        defaultModelId: event.target.value,
      },
    });
  };

  const onTimeoutChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: {
        ...jsonData,
        timeout: parseInt(event.target.value, 10) || 30,
      },
    });
  };

  const onAPIKeyChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      secureJsonData: {
        apiKey: event.target.value,
      },
    });
  };

  const onResetAPIKey = () => {
    onOptionsChange({
      ...options,
      secureJsonFields: {
        ...secureJsonFields,
        apiKey: false,
      },
      secureJsonData: {
        ...secureJsonData,
        apiKey: '',
      },
    });
  };

  return (
    <>
      <h3 className="page-heading">PyFlare Connection</h3>

      <InlineField label="URL" labelWidth={20} tooltip="PyFlare API endpoint URL">
        <Input
          width={40}
          value={jsonData.url || ''}
          onChange={onURLChange}
          placeholder="http://localhost:8080"
        />
      </InlineField>

      <InlineField
        label="API Key"
        labelWidth={20}
        tooltip="API key for authentication (optional)"
      >
        <SecretInput
          width={40}
          isConfigured={secureJsonFields?.apiKey}
          value={secureJsonData?.apiKey || ''}
          placeholder="Enter API key"
          onReset={onResetAPIKey}
          onChange={onAPIKeyChange}
        />
      </InlineField>

      <InlineField
        label="Timeout (seconds)"
        labelWidth={20}
        tooltip="Request timeout in seconds"
      >
        <Input
          width={40}
          type="number"
          value={jsonData.timeout || 30}
          onChange={onTimeoutChange}
          placeholder="30"
        />
      </InlineField>

      <h3 className="page-heading">Default Filters</h3>

      <InlineField
        label="Default Service"
        labelWidth={20}
        tooltip="Default service name to filter by"
      >
        <Input
          width={40}
          value={jsonData.defaultServiceName || ''}
          onChange={onDefaultServiceNameChange}
          placeholder="my-service"
        />
      </InlineField>

      <InlineField
        label="Default Model"
        labelWidth={20}
        tooltip="Default model ID to filter by"
      >
        <Input
          width={40}
          value={jsonData.defaultModelId || ''}
          onChange={onDefaultModelIdChange}
          placeholder="gpt-4"
        />
      </InlineField>
    </>
  );
}
