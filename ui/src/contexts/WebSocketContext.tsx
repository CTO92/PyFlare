/**
 * WebSocket Context
 * Provides WebSocket connection state and methods across the application
 */

import {
  createContext,
  useEffect,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import websocketService, {
  type ConnectionStatus,
  type WebSocketMessage,
} from '../services/websocket';

interface WebSocketContextType {
  status: ConnectionStatus;
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
  subscribe: (channel: string, callback: (message: WebSocketMessage) => void) => () => void;
  subscribeToAll: (callback: (message: WebSocketMessage) => void) => () => void;
  send: (message: Omit<WebSocketMessage, 'timestamp'>) => void;
}

export const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
  autoConnect?: boolean;
}

export function WebSocketProvider({
  children,
  autoConnect = true,
}: WebSocketProviderProps) {
  const [status, setStatus] = useState<ConnectionStatus>(websocketService.status);

  useEffect(() => {
    // Subscribe to status changes
    const unsubscribe = websocketService.subscribeToStatus(setStatus);

    // Auto-connect if enabled
    if (autoConnect) {
      websocketService.connect();
    }

    return () => {
      unsubscribe();
    };
  }, [autoConnect]);

  const connect = useCallback(() => {
    websocketService.connect();
  }, []);

  const disconnect = useCallback(() => {
    websocketService.disconnect();
  }, []);

  const subscribe = useCallback(
    (channel: string, callback: (message: WebSocketMessage) => void) => {
      return websocketService.subscribe(channel, callback);
    },
    []
  );

  const subscribeToAll = useCallback(
    (callback: (message: WebSocketMessage) => void) => {
      return websocketService.subscribeToAll(callback);
    },
    []
  );

  const send = useCallback(
    (message: Omit<WebSocketMessage, 'timestamp'>) => {
      websocketService.send(message);
    },
    []
  );

  return (
    <WebSocketContext.Provider
      value={{
        status,
        isConnected: status === 'connected',
        connect,
        disconnect,
        subscribe,
        subscribeToAll,
        send,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
}
