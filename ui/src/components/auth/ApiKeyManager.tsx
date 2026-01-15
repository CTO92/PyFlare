/**
 * API Key Manager Component
 * Generate, list, and revoke API keys
 */

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Key,
  Copy,
  Trash2,
  Plus,
  Check,
  AlertCircle,
  Eye,
  EyeOff,
  Loader2,
} from 'lucide-react';
import { format } from 'date-fns';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  lastUsedAt: string | null;
  expiresAt: string | null;
}

interface CreateApiKeyResponse {
  id: string;
  name: string;
  key: string; // Full key, only shown once
  prefix: string;
  createdAt: string;
}

async function fetchApiKeys(): Promise<ApiKey[]> {
  const response = await fetch(`${API_BASE}/auth/api-keys`, {
    credentials: 'include',
  });
  if (!response.ok) throw new Error('Failed to fetch API keys');
  const data = await response.json();
  return data.keys;
}

async function createApiKey(name: string): Promise<CreateApiKeyResponse> {
  const response = await fetch(`${API_BASE}/auth/api-keys`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw new Error('Failed to create API key');
  return response.json();
}

async function deleteApiKey(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/api-keys/${id}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (!response.ok) throw new Error('Failed to delete API key');
}

export default function ApiKeyManager() {
  const queryClient = useQueryClient();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<CreateApiKeyResponse | null>(null);
  const [showNewKey, setShowNewKey] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const { data: apiKeys = [], isLoading, error } = useQuery({
    queryKey: ['api-keys'],
    queryFn: fetchApiKeys,
  });

  const createMutation = useMutation({
    mutationFn: createApiKey,
    onSuccess: (data) => {
      setNewlyCreatedKey(data);
      setNewKeyName('');
      setShowCreateForm(false);
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (newKeyName.trim()) {
      createMutation.mutate(newKeyName.trim());
    }
  };

  const handleCopy = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      deleteMutation.mutate(id);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-red-50 p-4 text-red-700 dark:bg-red-900/20 dark:text-red-400">
        <AlertCircle className="h-5 w-5" />
        <span>Failed to load API keys</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Newly Created Key Alert */}
      {newlyCreatedKey && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
          <div className="flex items-start gap-3">
            <Check className="mt-0.5 h-5 w-5 flex-shrink-0 text-green-600 dark:text-green-400" />
            <div className="flex-1">
              <h4 className="font-medium text-green-800 dark:text-green-300">
                API Key Created Successfully
              </h4>
              <p className="mt-1 text-sm text-green-700 dark:text-green-400">
                Make sure to copy your API key now. You won&apos;t be able to see it again!
              </p>
              <div className="mt-3 flex items-center gap-2">
                <code className="flex-1 rounded bg-white px-3 py-2 font-mono text-sm dark:bg-gray-800">
                  {showNewKey ? newlyCreatedKey.key : `${newlyCreatedKey.prefix}${'*'.repeat(32)}`}
                </code>
                <button
                  onClick={() => setShowNewKey(!showNewKey)}
                  className="btn-ghost p-2"
                  title={showNewKey ? 'Hide key' : 'Show key'}
                >
                  {showNewKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
                <button
                  onClick={() => handleCopy(newlyCreatedKey.key, 'new')}
                  className="btn-ghost p-2"
                  title="Copy to clipboard"
                >
                  {copiedId === 'new' ? (
                    <Check className="h-4 w-4 text-green-600" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </button>
              </div>
              <button
                onClick={() => setNewlyCreatedKey(null)}
                className="mt-3 text-sm text-green-700 underline hover:no-underline dark:text-green-400"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">API Keys</h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Manage API keys for programmatic access to PyFlare
          </p>
        </div>
        {!showCreateForm && (
          <button
            onClick={() => setShowCreateForm(true)}
            className="btn-primary"
          >
            <Plus className="mr-2 h-4 w-4" />
            Create API Key
          </button>
        )}
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <form onSubmit={handleCreate} className="card p-4">
          <h4 className="mb-3 font-medium text-gray-900 dark:text-white">
            Create New API Key
          </h4>
          <div className="flex gap-3">
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              className="input flex-1"
              placeholder="Key name (e.g., Production Server)"
              autoFocus
            />
            <button
              type="submit"
              disabled={createMutation.isPending || !newKeyName.trim()}
              className="btn-primary"
            >
              {createMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                'Create'
              )}
            </button>
            <button
              type="button"
              onClick={() => setShowCreateForm(false)}
              className="btn-secondary"
            >
              Cancel
            </button>
          </div>
          {createMutation.error && (
            <p className="mt-2 text-sm text-red-600 dark:text-red-400">
              {createMutation.error.message}
            </p>
          )}
        </form>
      )}

      {/* API Keys List */}
      {apiKeys.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center dark:border-gray-700">
          <Key className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-white">
            No API keys yet
          </h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Create an API key to get started with programmatic access
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Key
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Created
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Last Used
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-900">
              {apiKeys.map((key) => (
                <tr key={key.id}>
                  <td className="whitespace-nowrap px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Key className="h-4 w-4 text-gray-400" />
                      <span className="font-medium text-gray-900 dark:text-white">
                        {key.name}
                      </span>
                    </div>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3">
                    <code className="text-sm text-gray-600 dark:text-gray-400">
                      {key.prefix}...
                    </code>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                    {format(new Date(key.createdAt), 'MMM d, yyyy')}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                    {key.lastUsedAt
                      ? format(new Date(key.lastUsedAt), 'MMM d, yyyy')
                      : 'Never'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-right">
                    <button
                      onClick={() => handleDelete(key.id)}
                      disabled={deleteMutation.isPending}
                      className="btn-ghost p-2 text-red-600 hover:bg-red-50 hover:text-red-700 dark:text-red-400 dark:hover:bg-red-900/20"
                      title="Delete API key"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
