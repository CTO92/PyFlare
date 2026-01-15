/**
 * Header Component
 * Top navigation bar with search, notifications, theme toggle, and user menu
 */

import { useState } from 'react';
import { Search, Sun, Moon, Menu } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import Breadcrumbs from './Breadcrumbs';
import UserMenu from './UserMenu';
import NotificationCenter from '../NotificationCenter';

interface HeaderProps {
  onMenuToggle?: () => void;
  showMenuButton?: boolean;
}

export default function Header({ onMenuToggle, showMenuButton = false }: HeaderProps) {
  const { toggleTheme, isDark } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchFocused, setSearchFocused] = useState(false);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Navigate to search results or filter traces
      window.location.href = `/traces?q=${encodeURIComponent(searchQuery)}`;
    }
  };

  return (
    <header className="sticky top-0 z-40 flex h-16 items-center gap-4 border-b border-gray-200 bg-white px-4 dark:border-gray-800 dark:bg-gray-900 lg:px-6">
      {/* Mobile Menu Button */}
      {showMenuButton && (
        <button
          onClick={onMenuToggle}
          className="btn-ghost p-2 lg:hidden"
          aria-label="Toggle menu"
        >
          <Menu className="h-5 w-5" />
        </button>
      )}

      {/* Breadcrumbs */}
      <div className="hidden flex-1 md:block">
        <Breadcrumbs />
      </div>

      {/* Search Bar */}
      <form
        onSubmit={handleSearch}
        className={`relative flex-1 transition-all duration-200 md:max-w-md ${
          searchFocused ? 'md:max-w-lg' : ''
        }`}
      >
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
            placeholder="Search traces, models, users..."
            className="input h-9 w-full pl-9 pr-12 text-sm"
          />
          <kbd className="pointer-events-none absolute right-3 top-1/2 hidden -translate-y-1/2 rounded border border-gray-200 bg-gray-100 px-1.5 py-0.5 text-xs text-gray-500 dark:border-gray-700 dark:bg-gray-800 sm:block">
            /
          </kbd>
        </div>
      </form>

      {/* Right Side Actions */}
      <div className="flex items-center gap-2">
        {/* Notification Center */}
        <NotificationCenter />

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="btn-ghost rounded-full p-2"
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? (
            <Sun className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          ) : (
            <Moon className="h-5 w-5 text-gray-500" />
          )}
        </button>

        {/* User Menu */}
        <UserMenu />
      </div>
    </header>
  );
}
