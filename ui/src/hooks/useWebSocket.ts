/**
 * useWebSocket Hook
 * Access WebSocket connection state and subscribe to channels
 */

import { useContext, useEffect, useState, useCallback } from 'react';
import {
  WebSocketContext,
} from '../contexts/WebSocketContext';
import type { ConnectionStatus, WebSocketMessage } from '../services/websocket';

export interface UseWebSocketReturn {
  status: ConnectionStatus;
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
  subscribe: (channel: string, callback: (message: WebSocketMessage) => void) => () => void;
  subscribeToAll: (callback: (message: WebSocketMessage) => void) => () => void;
  send: (message: Omit<WebSocketMessage, 'timestamp'>) => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const context = useContext(WebSocketContext);

  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }

  return context;
}

/**
 * Hook to subscribe to a specific WebSocket channel
 */
export function useChannel<T = unknown>(
  channel: string,
  onMessage?: (data: T, message: WebSocketMessage) => void
): {
  lastMessage: WebSocketMessage | null;
  data: T | null;
  isConnected: boolean;
} {
  const { subscribe, isConnected } = useWebSocket();
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [data, setData] = useState<T | null>(null);

  useEffect(() => {
    const unsubscribe = subscribe(channel, (message) => {
      setLastMessage(message);
      setData(message.data as T);
      onMessage?.(message.data as T, message);
    });

    return unsubscribe;
  }, [channel, subscribe, onMessage]);

  return { lastMessage, data, isConnected };
}

/**
 * Hook to subscribe to real-time trace updates
 */
export function useTraceStream(
  modelId?: string,
  onTrace?: (trace: unknown) => void
): {
  traces: unknown[];
  isConnected: boolean;
  clearTraces: () => void;
} {
  const { subscribe, isConnected } = useWebSocket();
  const [traces, setTraces] = useState<unknown[]>([]);

  const clearTraces = useCallback(() => {
    setTraces([]);
  }, []);

  useEffect(() => {
    const channel = modelId ? `traces:${modelId}` : 'traces:*';

    const unsubscribe = subscribe(channel, (message) => {
      if (message.event === 'trace.created' || message.event === 'trace.updated') {
        setTraces((prev) => [message.data, ...prev].slice(0, 100)); // Keep last 100
        onTrace?.(message.data);
      }
    });

    return unsubscribe;
  }, [modelId, subscribe, onTrace]);

  return { traces, isConnected, clearTraces };
}

/**
 * Hook to subscribe to real-time alert updates
 */
export function useAlertStream(
  onAlert?: (alert: unknown) => void
): {
  alerts: unknown[];
  isConnected: boolean;
  clearAlerts: () => void;
} {
  const { subscribe, isConnected } = useWebSocket();
  const [alerts, setAlerts] = useState<unknown[]>([]);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  useEffect(() => {
    const unsubscribe = subscribe('alerts:*', (message) => {
      if (message.event === 'alert.fired') {
        setAlerts((prev) => [message.data, ...prev].slice(0, 50));
        onAlert?.(message.data);
      } else if (message.event === 'alert.resolved') {
        setAlerts((prev) =>
          prev.map((a: { id?: string }) =>
            (a as { id: string }).id === (message.data as { id: string }).id
              ? message.data
              : a
          )
        );
      }
    });

    return unsubscribe;
  }, [subscribe, onAlert]);

  return { alerts, isConnected, clearAlerts };
}

/**
 * Hook to subscribe to real-time drift updates
 */
export function useDriftStream(
  modelId?: string,
  onDrift?: (drift: unknown) => void
): {
  driftScore: number | null;
  lastUpdate: string | null;
  isConnected: boolean;
} {
  const { subscribe, isConnected } = useWebSocket();
  const [driftScore, setDriftScore] = useState<number | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  useEffect(() => {
    const channel = modelId ? `drift:${modelId}` : 'drift:*';

    const unsubscribe = subscribe(channel, (message) => {
      if (message.event === 'drift.updated') {
        const data = message.data as { score?: number };
        setDriftScore(data.score ?? null);
        setLastUpdate(message.timestamp);
        onDrift?.(message.data);
      }
    });

    return unsubscribe;
  }, [modelId, subscribe, onDrift]);

  return { driftScore, lastUpdate, isConnected };
}

export default useWebSocket;
