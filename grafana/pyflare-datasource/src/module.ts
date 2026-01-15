import { DataSourcePlugin } from '@grafana/data';
import { PyFlareDataSource } from './datasource';
import { ConfigEditor } from './ConfigEditor';
import { QueryEditor } from './QueryEditor';
import { PyFlareQuery, PyFlareDataSourceOptions } from './types';

export const plugin = new DataSourcePlugin<PyFlareDataSource, PyFlareQuery, PyFlareDataSourceOptions>(
  PyFlareDataSource
)
  .setConfigEditor(ConfigEditor)
  .setQueryEditor(QueryEditor);
