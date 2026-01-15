/**
 * Auth Context
 * Provides authentication state and methods across the application
 * Supports both JWT token and API key authentication
 *
 * Security Features:
 * - Prefers httpOnly cookies for token storage (server-side)
 * - Falls back to encrypted sessionStorage when cookies unavailable
 * - Implements CSRF protection for state-changing requests
 * - Uses short-lived access tokens with refresh
 */

import {
  createContext,
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { secureStorage, isSecureStorageAvailable } from '../utils/secureStorage';
import { clearCSRFToken, getCSRFHeaders } from '../utils/csrf';

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  avatarUrl?: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  loginWithApiKey: (apiKey: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshAuth: () => Promise<void>;
  getAuthHeaders: () => Promise<Record<string, string>>;
}

export const AuthContext = createContext<AuthContextType | null>(null);

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';
const STORAGE_KEYS = {
  accessToken: 'access_token',
  refreshToken: 'refresh_token',
  tokenExpiry: 'token_expiry',
  apiKey: 'api_key',
  user: 'user',
};

// Token refresh threshold (5 minutes before expiry)
const REFRESH_THRESHOLD_MS = 5 * 60 * 1000;

// Access token TTL for secure storage (15 minutes)
const ACCESS_TOKEN_TTL_MS = 15 * 60 * 1000;

// Refresh token TTL for secure storage (7 days)
const REFRESH_TOKEN_TTL_MS = 7 * 24 * 60 * 60 * 1000;

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [tokenExpiry, setTokenExpiry] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [useHttpOnlyCookies, setUseHttpOnlyCookies] = useState(false);

  const isAuthenticated = !!(user && (accessToken || apiKey || useHttpOnlyCookies));

  /**
   * Clear all auth state securely
   */
  const clearAuth = useCallback(async () => {
    setUser(null);
    setAccessToken(null);
    setApiKey(null);
    setTokenExpiry(null);
    setUseHttpOnlyCookies(false);

    // Clear secure storage
    secureStorage.clear();

    // Clear CSRF token
    clearCSRFToken();

    // Notify server to clear httpOnly cookies
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...getCSRFHeaders(),
        },
      });
    } catch {
      // Ignore logout errors - client state is already cleared
    }
  }, []);

  /**
   * Securely store tokens
   */
  const storeTokens = useCallback(async (tokens: AuthTokens) => {
    if (isSecureStorageAvailable()) {
      await secureStorage.setItem(
        STORAGE_KEYS.accessToken,
        tokens.accessToken,
        ACCESS_TOKEN_TTL_MS
      );
      await secureStorage.setItem(
        STORAGE_KEYS.refreshToken,
        tokens.refreshToken,
        REFRESH_TOKEN_TTL_MS
      );
      await secureStorage.setItem(
        STORAGE_KEYS.tokenExpiry,
        tokens.expiresAt.toString(),
        REFRESH_TOKEN_TTL_MS
      );
    }
    setAccessToken(tokens.accessToken);
    setTokenExpiry(tokens.expiresAt);
  }, []);

  /**
   * Securely store user profile
   */
  const storeUser = useCallback(async (userProfile: User) => {
    if (isSecureStorageAvailable()) {
      await secureStorage.setItem(
        STORAGE_KEYS.user,
        JSON.stringify(userProfile),
        REFRESH_TOKEN_TTL_MS
      );
    }
    setUser(userProfile);
  }, []);

  /**
   * Fetch user profile
   */
  const fetchUserProfile = useCallback(async (token: string): Promise<User> => {
    const headers: Record<string, string> = {
      ...getCSRFHeaders(),
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE}/auth/profile`, {
      headers,
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user profile');
    }

    return response.json();
  }, []);

  /**
   * Refresh access token
   */
  const refreshAccessToken = useCallback(async (): Promise<AuthTokens | null> => {
    // First try httpOnly cookie refresh (server handles token)
    const cookieResponse = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...getCSRFHeaders(),
      },
    });

    if (cookieResponse.ok) {
      const data = await cookieResponse.json();
      // Server might return tokens or just set new cookies
      if (data.accessToken) {
        return data;
      }
      // httpOnly cookie mode - tokens are in cookies
      setUseHttpOnlyCookies(true);
      return null;
    }

    // Fall back to refresh token from secure storage
    const storedRefreshToken = await secureStorage.getItem(STORAGE_KEYS.refreshToken);
    if (!storedRefreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getCSRFHeaders(),
      },
      credentials: 'include',
      body: JSON.stringify({ refreshToken: storedRefreshToken }),
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    return response.json();
  }, []);

  /**
   * Login with email/password
   */
  const login = useCallback(async (credentials: LoginCredentials) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getCSRFHeaders(),
        },
        credentials: 'include', // Allow server to set httpOnly cookies
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'Login failed');
      }

      const data = await response.json();

      // Check if server uses httpOnly cookies
      if (data.useHttpOnlyCookies) {
        setUseHttpOnlyCookies(true);
        // Tokens are in httpOnly cookies, just store user
        if (data.user) {
          await storeUser(data.user);
        } else {
          // Fetch user profile separately
          const userProfile = await fetchUserProfile('');
          await storeUser(userProfile);
        }
      } else if (data.accessToken) {
        // Server returned tokens - store securely
        const authTokens: AuthTokens = {
          accessToken: data.accessToken,
          refreshToken: data.refreshToken,
          expiresAt: data.expiresAt || Date.now() + ACCESS_TOKEN_TTL_MS,
        };
        await storeTokens(authTokens);

        const userProfile = await fetchUserProfile(authTokens.accessToken);
        await storeUser(userProfile);
      } else {
        throw new Error('Invalid server response');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchUserProfile, storeTokens, storeUser]);

  /**
   * Login with API key
   */
  const loginWithApiKey = useCallback(async (key: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/auth/verify-api-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': key,
          ...getCSRFHeaders(),
        },
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Invalid API key');
      }

      const data = await response.json();

      // Store API key securely (with shorter TTL for security)
      if (isSecureStorageAvailable()) {
        // API keys get 24 hour TTL in secure storage
        await secureStorage.setItem(
          STORAGE_KEYS.apiKey,
          key,
          24 * 60 * 60 * 1000
        );
      }
      setApiKey(key);

      if (data.user) {
        await storeUser(data.user);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'API key verification failed';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [storeUser]);

  /**
   * Logout
   */
  const logout = useCallback(async () => {
    await clearAuth();
    setError(null);
  }, [clearAuth]);

  /**
   * Refresh auth (token refresh)
   */
  const refreshAuth = useCallback(async () => {
    try {
      const newTokens = await refreshAccessToken();
      if (newTokens) {
        await storeTokens(newTokens);
      }
    } catch {
      // Refresh failed, clear auth state
      await clearAuth();
    }
  }, [refreshAccessToken, storeTokens, clearAuth]);

  /**
   * Get authorization headers for API requests
   */
  const getAuthHeaders = useCallback(async (): Promise<Record<string, string>> => {
    const headers: Record<string, string> = {
      ...getCSRFHeaders(),
    };

    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    } else if (accessToken) {
      // Check if token needs refresh
      if (tokenExpiry && tokenExpiry - Date.now() < REFRESH_THRESHOLD_MS) {
        try {
          const newTokens = await refreshAccessToken();
          if (newTokens) {
            await storeTokens(newTokens);
            headers['Authorization'] = `Bearer ${newTokens.accessToken}`;
            return headers;
          }
        } catch {
          // Continue with existing token
        }
      }
      headers['Authorization'] = `Bearer ${accessToken}`;
    }
    // If using httpOnly cookies, no Authorization header needed

    return headers;
  }, [apiKey, accessToken, tokenExpiry, refreshAccessToken, storeTokens]);

  /**
   * Initialize auth state from secure storage
   */
  useEffect(() => {
    const initAuth = async () => {
      try {
        // First check if we have valid httpOnly cookies by calling profile
        try {
          const response = await fetch(`${API_BASE}/auth/profile`, {
            credentials: 'include',
            headers: getCSRFHeaders(),
          });

          if (response.ok) {
            const userProfile = await response.json();
            setUser(userProfile);
            setUseHttpOnlyCookies(true);
            setIsLoading(false);
            return;
          }
        } catch {
          // No valid cookies, continue with stored tokens
        }

        // Check for stored tokens in secure storage
        if (isSecureStorageAvailable()) {
          const storedAccessToken = await secureStorage.getItem(STORAGE_KEYS.accessToken);
          const storedExpiry = await secureStorage.getItem(STORAGE_KEYS.tokenExpiry);
          const storedApiKey = await secureStorage.getItem(STORAGE_KEYS.apiKey);
          const storedUser = await secureStorage.getItem(STORAGE_KEYS.user);

          if (storedAccessToken && storedExpiry) {
            const expiry = parseInt(storedExpiry, 10);

            // Check if token needs refresh
            if (expiry - Date.now() < REFRESH_THRESHOLD_MS) {
              try {
                const newTokens = await refreshAccessToken();
                if (newTokens) {
                  await storeTokens(newTokens);
                }
              } catch {
                // Refresh failed, clear stored data
                await clearAuth();
                setIsLoading(false);
                return;
              }
            } else {
              setAccessToken(storedAccessToken);
              setTokenExpiry(expiry);
            }

            if (storedUser) {
              try {
                setUser(JSON.parse(storedUser));
              } catch {
                // Invalid stored user, fetch fresh
                const userProfile = await fetchUserProfile(storedAccessToken);
                await storeUser(userProfile);
              }
            }
          } else if (storedApiKey) {
            setApiKey(storedApiKey);
            if (storedUser) {
              try {
                setUser(JSON.parse(storedUser));
              } catch {
                // Invalid stored user
              }
            }
          }
        }
      } catch {
        // Clear invalid stored data
        await clearAuth();
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, [clearAuth, refreshAccessToken, storeTokens, storeUser, fetchUserProfile]);

  /**
   * Auto-refresh token before expiry
   */
  useEffect(() => {
    if (!tokenExpiry || useHttpOnlyCookies) return;

    const timeUntilRefresh = tokenExpiry - Date.now() - REFRESH_THRESHOLD_MS;
    if (timeUntilRefresh <= 0) {
      refreshAuth();
      return;
    }

    const timer = setTimeout(refreshAuth, timeUntilRefresh);
    return () => clearTimeout(timer);
  }, [tokenExpiry, refreshAuth, useHttpOnlyCookies]);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated,
        isLoading,
        error,
        login,
        loginWithApiKey,
        logout,
        refreshAuth,
        getAuthHeaders,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
