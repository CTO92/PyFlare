/**
 * Sidebar Component
 * Navigation sidebar with collapsible state
 */

import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';
import {
  LayoutDashboard,
  Search,
  TrendingUp,
  DollarSign,
  Bell,
  Settings,
  Flame,
  Brain,
  FileSearch,
  ChevronLeft,
  ChevronRight,
  X,
} from 'lucide-react';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Traces', href: '/traces', icon: Search },
  { name: 'Drift', href: '/drift', icon: TrendingUp },
  { name: 'Costs', href: '/costs', icon: DollarSign },
  { name: 'Alerts', href: '/alerts', icon: Bell },
  { name: 'Intelligence', href: '/intelligence', icon: Brain },
  { name: 'RCA', href: '/rca', icon: FileSearch },
  { name: 'Settings', href: '/settings', icon: Settings },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const COLLAPSED_KEY = 'pyflare-sidebar-collapsed';

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(() => {
    const stored = localStorage.getItem(COLLAPSED_KEY);
    return stored === 'true';
  });

  useEffect(() => {
    localStorage.setItem(COLLAPSED_KEY, String(isCollapsed));
  }, [isCollapsed]);

  const toggleCollapsed = () => setIsCollapsed(!isCollapsed);

  const sidebarContent = (
    <div className="flex h-full flex-col">
      {/* Logo */}
      <div
        className={clsx(
          'flex h-16 items-center border-b border-gray-200 dark:border-gray-800',
          isCollapsed ? 'justify-center px-2' : 'gap-2 px-6'
        )}
      >
        <Flame className="h-8 w-8 flex-shrink-0 text-pyflare-500" />
        {!isCollapsed && (
          <span className="text-xl font-bold text-gray-900 dark:text-white">
            PyFlare
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => {
          const isActive =
            location.pathname === item.href ||
            (item.href !== '/' && location.pathname.startsWith(item.href));

          return (
            <Link
              key={item.name}
              to={item.href}
              onClick={() => {
                // Close mobile sidebar on navigation
                if (window.innerWidth < 1024) {
                  onClose();
                }
              }}
              className={clsx(
                'flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isCollapsed ? 'justify-center' : 'gap-3',
                isActive
                  ? 'bg-pyflare-50 text-pyflare-600 dark:bg-pyflare-900/20 dark:text-pyflare-400'
                  : 'text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'
              )}
              title={isCollapsed ? item.name : undefined}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!isCollapsed && item.name}
            </Link>
          );
        })}
      </nav>

      {/* Collapse Toggle (Desktop only) */}
      <div className="hidden border-t border-gray-200 p-2 dark:border-gray-800 lg:block">
        <button
          onClick={toggleCollapsed}
          className="flex w-full items-center justify-center rounded-md p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <ChevronLeft className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Footer */}
      <div
        className={clsx(
          'border-t border-gray-200 p-4 dark:border-gray-800',
          isCollapsed && 'hidden'
        )}
      >
        <p className="text-xs text-gray-500">PyFlare v0.1.0</p>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        className={clsx(
          'hidden border-r border-gray-200 bg-white transition-all duration-300 dark:border-gray-800 dark:bg-gray-900 lg:block',
          isCollapsed ? 'w-16' : 'w-64'
        )}
      >
        {sidebarContent}
      </aside>

      {/* Mobile Sidebar Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/50 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Mobile Sidebar */}
      <aside
        className={clsx(
          'fixed inset-y-0 left-0 z-50 w-64 transform border-r border-gray-200 bg-white transition-transform duration-300 dark:border-gray-800 dark:bg-gray-900 lg:hidden',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute right-2 top-4 p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
        >
          <X className="h-5 w-5" />
        </button>
        {sidebarContent}
      </aside>
    </>
  );
}
