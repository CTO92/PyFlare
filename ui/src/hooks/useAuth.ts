/**
 * useAuth Hook
 * Access authentication state and methods
 */

import { useContext } from 'react';
import { AuthContext, type User, type LoginCredentials } from '../contexts/AuthContext';

export interface UseAuthReturn {
  user: User | null;
  apiKey: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  loginWithApiKey: (apiKey: string) => Promise<void>;
  logout: () => void;
  setApiKey: (key: string | null) => void;
  refreshAuth: () => Promise<void>;
}

export function useAuth(): UseAuthReturn {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
}

export default useAuth;
