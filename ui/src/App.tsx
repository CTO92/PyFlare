/**
 * PyFlare Application Root
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Traces from './pages/Traces';
import TraceDetail from './pages/TraceDetail';
import Drift from './pages/Drift';
import Costs from './pages/Costs';
import Alerts from './pages/Alerts';
import Settings from './pages/Settings';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public Routes */}
            <Route path="/login" element={<Login />} />

            {/* Protected Routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <WebSocketProvider>
                    <Layout />
                  </WebSocketProvider>
                </ProtectedRoute>
              }
            >
              <Route index element={<Dashboard />} />
              <Route path="traces" element={<Traces />} />
              <Route path="traces/:traceId" element={<TraceDetail />} />
              <Route path="drift" element={<Drift />} />
              <Route path="costs" element={<Costs />} />
              <Route path="alerts" element={<Alerts />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
