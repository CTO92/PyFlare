/**
 * Breadcrumbs Component
 * Dynamic breadcrumb navigation based on current route
 */

import { Link, useLocation, useParams } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';

interface BreadcrumbItem {
  label: string;
  href?: string;
}

const routeLabels: Record<string, string> = {
  traces: 'Traces',
  drift: 'Drift Detection',
  costs: 'Cost Analytics',
  alerts: 'Alerts',
  settings: 'Settings',
  intelligence: 'Intelligence',
  rca: 'Root Cause Analysis',
  login: 'Login',
  compare: 'Compare',
};

function generateBreadcrumbs(pathname: string, params: Record<string, string | undefined>): BreadcrumbItem[] {
  const segments = pathname.split('/').filter(Boolean);
  const breadcrumbs: BreadcrumbItem[] = [];

  let currentPath = '';

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    currentPath += `/${segment}`;

    // Check if this is a dynamic segment (like traceId)
    if (params.traceId && segment === params.traceId) {
      breadcrumbs.push({
        label: `Trace: ${segment.slice(0, 8)}...`,
      });
    } else if (routeLabels[segment]) {
      breadcrumbs.push({
        label: routeLabels[segment],
        href: i < segments.length - 1 ? currentPath : undefined,
      });
    } else {
      breadcrumbs.push({
        label: segment.charAt(0).toUpperCase() + segment.slice(1),
        href: i < segments.length - 1 ? currentPath : undefined,
      });
    }
  }

  return breadcrumbs;
}

export default function Breadcrumbs() {
  const location = useLocation();
  const params = useParams();

  // Don't show breadcrumbs on home page
  if (location.pathname === '/') {
    return (
      <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
        <Home className="h-4 w-4" />
        <span className="ml-2">Dashboard</span>
      </div>
    );
  }

  const breadcrumbs = generateBreadcrumbs(location.pathname, params);

  return (
    <nav className="flex items-center text-sm" aria-label="Breadcrumb">
      <ol className="flex items-center space-x-1">
        {/* Home Link */}
        <li>
          <Link
            to="/"
            className="flex items-center text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <Home className="h-4 w-4" />
          </Link>
        </li>

        {/* Breadcrumb Items */}
        {breadcrumbs.map((crumb, index) => (
          <li key={index} className="flex items-center">
            <ChevronRight className="mx-1 h-4 w-4 text-gray-400" />
            {crumb.href ? (
              <Link
                to={crumb.href}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                {crumb.label}
              </Link>
            ) : (
              <span className="font-medium text-gray-900 dark:text-white">
                {crumb.label}
              </span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
