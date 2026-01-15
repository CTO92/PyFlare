/**
 * User Menu Component
 * Dropdown menu for user actions
 */

import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { User, Settings, LogOut, Key, HelpCircle } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

export default function UserMenu() {
  const { user, logout } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Close menu on escape key
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  const handleLogout = () => {
    setIsOpen(false);
    logout();
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-full p-1 hover:bg-gray-100 dark:hover:bg-gray-800"
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        {user?.avatarUrl ? (
          <img
            src={user.avatarUrl}
            alt={user.name}
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-pyflare-100 dark:bg-pyflare-900">
            <User className="h-4 w-4 text-pyflare-600 dark:text-pyflare-400" />
          </div>
        )}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 origin-top-right rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-gray-700 dark:bg-gray-800">
          {/* User Info */}
          <div className="border-b border-gray-200 px-4 py-3 dark:border-gray-700">
            <p className="text-sm font-medium text-gray-900 dark:text-white">
              {user?.name || 'User'}
            </p>
            <p className="truncate text-xs text-gray-500 dark:text-gray-400">
              {user?.email || 'API Key User'}
            </p>
            {user?.role && (
              <span className="mt-1 inline-block rounded bg-gray-100 px-2 py-0.5 text-xs capitalize text-gray-600 dark:bg-gray-700 dark:text-gray-300">
                {user.role}
              </span>
            )}
          </div>

          {/* Menu Items */}
          <div className="py-1">
            <Link
              to="/settings"
              onClick={() => setIsOpen(false)}
              className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
            >
              <Settings className="h-4 w-4" />
              Settings
            </Link>

            <Link
              to="/settings#api-keys"
              onClick={() => setIsOpen(false)}
              className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
            >
              <Key className="h-4 w-4" />
              API Keys
            </Link>

            <a
              href="https://docs.pyflare.io"
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setIsOpen(false)}
              className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
            >
              <HelpCircle className="h-4 w-4" />
              Documentation
            </a>
          </div>

          {/* Logout */}
          <div className="border-t border-gray-200 py-1 dark:border-gray-700">
            <button
              onClick={handleLogout}
              className="flex w-full items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20"
            >
              <LogOut className="h-4 w-4" />
              Sign out
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
