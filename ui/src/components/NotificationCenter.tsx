/**
 * Notification Center Component
 * Real-time notifications dropdown
 */

import { useState, useRef, useEffect } from 'react';
import { Bell, X, AlertTriangle, TrendingUp, DollarSign, Info, Check } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

export interface Notification {
  id: string;
  type: 'alert' | 'drift' | 'cost' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  link?: string;
}

// Mock notifications - in production, these would come from WebSocket or API
const mockNotifications: Notification[] = [
  {
    id: '1',
    type: 'alert',
    title: 'High Error Rate Detected',
    message: 'Model gpt-4 error rate exceeded 5% threshold',
    timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
    read: false,
    link: '/alerts',
  },
  {
    id: '2',
    type: 'drift',
    title: 'Feature Drift Detected',
    message: 'Embedding drift score 0.35 for model claude-3',
    timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    read: false,
    link: '/drift',
  },
  {
    id: '3',
    type: 'cost',
    title: 'Budget Warning',
    message: 'Monthly budget at 80% utilization',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    read: true,
    link: '/costs',
  },
];

const typeIcons = {
  alert: AlertTriangle,
  drift: TrendingUp,
  cost: DollarSign,
  info: Info,
  success: Check,
};

const typeColors = {
  alert: 'text-red-500 bg-red-50 dark:bg-red-900/20',
  drift: 'text-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
  cost: 'text-blue-500 bg-blue-50 dark:bg-blue-900/20',
  info: 'text-gray-500 bg-gray-50 dark:bg-gray-800',
  success: 'text-green-500 bg-green-50 dark:bg-green-900/20',
};

export default function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);
  const menuRef = useRef<HTMLDivElement>(null);

  const unreadCount = notifications.filter((n) => !n.read).length;

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

  const handleMarkAsRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const handleMarkAllAsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const handleDismiss = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const handleClearAll = () => {
    setNotifications([]);
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* Bell Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="btn-ghost relative rounded-full p-2"
        aria-label="Notifications"
      >
        <Bell className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        {unreadCount > 0 && (
          <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[10px] font-medium text-white">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 origin-top-right rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800 sm:w-96">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700">
            <h3 className="font-medium text-gray-900 dark:text-white">
              Notifications
            </h3>
            <div className="flex items-center gap-2">
              {unreadCount > 0 && (
                <button
                  onClick={handleMarkAllAsRead}
                  className="text-xs text-pyflare-600 hover:text-pyflare-700 dark:text-pyflare-400 dark:hover:text-pyflare-300"
                >
                  Mark all read
                </button>
              )}
              {notifications.length > 0 && (
                <button
                  onClick={handleClearAll}
                  className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  Clear all
                </button>
              )}
            </div>
          </div>

          {/* Notifications List */}
          <div className="max-h-96 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="px-4 py-8 text-center">
                <Bell className="mx-auto h-8 w-8 text-gray-400" />
                <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                  No notifications
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {notifications.map((notification) => {
                  const Icon = typeIcons[notification.type];

                  return (
                    <div
                      key={notification.id}
                      className={`relative px-4 py-3 transition-colors hover:bg-gray-50 dark:hover:bg-gray-700/50 ${
                        !notification.read ? 'bg-pyflare-50/50 dark:bg-pyflare-900/10' : ''
                      }`}
                    >
                      <div className="flex gap-3">
                        <div
                          className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${
                            typeColors[notification.type]
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-start justify-between gap-2">
                            <a
                              href={notification.link || '#'}
                              onClick={() => {
                                handleMarkAsRead(notification.id);
                                setIsOpen(false);
                              }}
                              className="font-medium text-gray-900 hover:text-pyflare-600 dark:text-white dark:hover:text-pyflare-400"
                            >
                              {notification.title}
                            </a>
                            <button
                              onClick={() => handleDismiss(notification.id)}
                              className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                          <p className="mt-0.5 text-sm text-gray-600 dark:text-gray-400">
                            {notification.message}
                          </p>
                          <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
                            {formatDistanceToNow(new Date(notification.timestamp), {
                              addSuffix: true,
                            })}
                          </p>
                        </div>
                      </div>
                      {!notification.read && (
                        <div className="absolute left-1.5 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-pyflare-500" />
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Footer */}
          {notifications.length > 0 && (
            <div className="border-t border-gray-200 px-4 py-2 dark:border-gray-700">
              <a
                href="/alerts"
                onClick={() => setIsOpen(false)}
                className="block text-center text-sm text-pyflare-600 hover:text-pyflare-700 dark:text-pyflare-400 dark:hover:text-pyflare-300"
              >
                View all alerts
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
