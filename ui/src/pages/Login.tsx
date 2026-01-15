/**
 * Login Page
 * Full-page login with branding
 */

import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Flame } from 'lucide-react';
import LoginForm from '../components/auth/LoginForm';
import { useAuth } from '../hooks/useAuth';
import { useTheme } from '../hooks/useTheme';

export default function Login() {
  const { isAuthenticated, isLoading } = useAuth();
  const { toggleTheme, isDark } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      const from = (location.state as { from?: string })?.from || '/';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate, location]);

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-pyflare-500 border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-gray-50 dark:bg-gray-900">
      {/* Theme Toggle */}
      <div className="absolute right-4 top-4">
        <button
          onClick={toggleTheme}
          className="btn-ghost rounded-full p-2"
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? (
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
              />
            </svg>
          ) : (
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
              />
            </svg>
          )}
        </button>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
        {/* Logo and Title */}
        <div className="mb-8 text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-pyflare-400 to-pyflare-600 shadow-lg">
            <Flame className="h-10 w-10 text-white" />
          </div>
          <h1 className="mt-4 text-3xl font-bold text-gray-900 dark:text-white">
            PyFlare
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            AI/ML Observability Platform
          </p>
        </div>

        {/* Login Card */}
        <div className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-8 shadow-lg dark:border-gray-700 dark:bg-gray-800">
          <h2 className="mb-6 text-center text-xl font-semibold text-gray-900 dark:text-white">
            Sign in to your account
          </h2>
          <LoginForm />
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>
            Don&apos;t have an account?{' '}
            <a
              href="https://docs.pyflare.io/getting-started"
              className="font-medium text-pyflare-600 hover:text-pyflare-500 dark:text-pyflare-400"
              target="_blank"
              rel="noopener noreferrer"
            >
              Get started with PyFlare
            </a>
          </p>
        </div>
      </div>

      {/* Version Footer */}
      <footer className="py-4 text-center text-xs text-gray-400 dark:text-gray-600">
        PyFlare v0.1.0
      </footer>
    </div>
  );
}
