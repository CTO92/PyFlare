/**
 * useTheme Hook
 * Access theme state and toggle functionality
 */

import { useContext } from 'react';
import { ThemeContext, type Theme } from '../contexts/ThemeContext';

export interface UseThemeReturn {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  isDark: boolean;
}

export function useTheme(): UseThemeReturn {
  const context = useContext(ThemeContext);

  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }

  return {
    ...context,
    isDark: context.resolvedTheme === 'dark',
  };
}

export default useTheme;
