/**
 * Main Layout Component
 * Application shell with sidebar, header, and main content area
 */

import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Header from './layout/Header';
import Sidebar from './layout/Sidebar';

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main Content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <Header
          showMenuButton
          onMenuToggle={() => setSidebarOpen(true)}
        />

        {/* Page Content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
