/**
 * Login Form Component
 * Supports email/password and API key authentication
 */

import { useState } from 'react';
import { Eye, EyeOff, Key, Mail, Lock, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

type AuthMode = 'credentials' | 'apikey';

export default function LoginForm() {
  const { login, loginWithApiKey, isLoading, error } = useAuth();

  const [authMode, setAuthMode] = useState<AuthMode>('credentials');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleCredentialsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (!email.trim()) {
      setLocalError('Email is required');
      return;
    }

    if (!password) {
      setLocalError('Password is required');
      return;
    }

    try {
      await login({ email: email.trim(), password });
    } catch {
      // Error is handled by AuthContext
    }
  };

  const handleApiKeySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (!apiKey.trim()) {
      setLocalError('API key is required');
      return;
    }

    try {
      await loginWithApiKey(apiKey.trim());
    } catch {
      // Error is handled by AuthContext
    }
  };

  const displayError = localError || error;

  return (
    <div className="w-full max-w-md">
      {/* Auth Mode Tabs */}
      <div className="mb-6 flex rounded-lg bg-gray-100 p-1 dark:bg-gray-800">
        <button
          type="button"
          onClick={() => setAuthMode('credentials')}
          className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            authMode === 'credentials'
              ? 'bg-white text-gray-900 shadow-sm dark:bg-gray-700 dark:text-white'
              : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'
          }`}
        >
          <Mail className="h-4 w-4" />
          Email
        </button>
        <button
          type="button"
          onClick={() => setAuthMode('apikey')}
          className={`flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            authMode === 'apikey'
              ? 'bg-white text-gray-900 shadow-sm dark:bg-gray-700 dark:text-white'
              : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white'
          }`}
        >
          <Key className="h-4 w-4" />
          API Key
        </button>
      </div>

      {/* Error Display */}
      {displayError && (
        <div className="mb-4 flex items-center gap-2 rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-400">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Credentials Form */}
      {authMode === 'credentials' && (
        <form onSubmit={handleCredentialsSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
            >
              Email
            </label>
            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <Mail className="h-4 w-4 text-gray-400" />
              </div>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input pl-10"
                placeholder="you@example.com"
                autoComplete="email"
                disabled={isLoading}
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="password"
              className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
            >
              Password
            </label>
            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <Lock className="h-4 w-4 text-gray-400" />
              </div>
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input pl-10 pr-10"
                placeholder="Enter your password"
                autoComplete="current-password"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Signing in...
              </>
            ) : (
              'Sign in'
            )}
          </button>
        </form>
      )}

      {/* API Key Form */}
      {authMode === 'apikey' && (
        <form onSubmit={handleApiKeySubmit} className="space-y-4">
          <div>
            <label
              htmlFor="apikey"
              className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
            >
              API Key
            </label>
            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <Key className="h-4 w-4 text-gray-400" />
              </div>
              <input
                id="apikey"
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="input pl-10 pr-10 font-mono"
                placeholder="pf_xxxxxxxxxxxxx"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Generate an API key from Settings → API Keys
            </p>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Verifying...
              </>
            ) : (
              'Connect with API Key'
            )}
          </button>
        </form>
      )}
    </div>
  );
}
