/**
 * WebSocket Service
 * Manages WebSocket connections with automatic reconnection and channel subscriptions
 *
 * Security Features:
 * - Authentication via token in URL or initial auth message
 * - Connection timeout for unauthenticated connections
 * - Secure reconnection with token refresh
 */

import { secureStorage } from '../utils/secureStorage';
import { getCSRFToken } from '../utils/csrf';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting' | 'authenticating';

export interface WebSocketMessage {
  channel: string;
  event: string;
  data: unknown;
  timestamp: string;
}

type MessageCallback = (message: WebSocketMessage) => void;
type AuthTokenProvider = () => Promise<string | null>;

interface WebSocketServiceConfig {
  url: string;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  authTimeout?: number;
}

const DEFAULT_CONFIG: Required<Omit<WebSocketServiceConfig, 'url'>> = {
  maxReconnectAttempts: 5,
  reconnectInterval: 3000,
  heartbeatInterval: 30000,
  authTimeout: 10000, // 10 seconds to authenticate
};

class WebSocketService {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketServiceConfig>;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private authTimer: ReturnType<typeof setTimeout> | null = null;
  private subscribers: Map<string, Set<MessageCallback>> = new Map();
  private globalSubscribers: Set<MessageCallback> = new Set();
  private statusSubscribers: Set<(status: ConnectionStatus) => void> = new Set();
  private _status: ConnectionStatus = 'disconnected';
  private pendingSubscriptions: Set<string> = new Set();
  private isAuthenticated = false;
  private authTokenProvider: AuthTokenProvider | null = null;

  constructor(config: WebSocketServiceConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  get status(): ConnectionStatus {
    return this._status;
  }

  /**
   * Set the authentication token provider
   * This function should return the current auth token or null
   */
  setAuthTokenProvider(provider: AuthTokenProvider): void {
    this.authTokenProvider = provider;
  }

  private setStatus(status: ConnectionStatus): void {
    this._status = status;
    this.statusSubscribers.forEach((callback) => callback(status));
  }

  /**
   * Connect to WebSocket with authentication
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.setStatus('connecting');
    this.isAuthenticated = false;

    try {
      // Get auth token if provider is set
      let authToken: string | null = null;
      if (this.authTokenProvider) {
        authToken = await this.authTokenProvider();
      }

      // Fallback: try to get token from secure storage
      if (!authToken) {
        authToken = await secureStorage.getItem('access_token');
      }

      // Build URL with auth token as query parameter (secure way for WebSocket)
      let wsUrl = this.config.url;
      if (authToken) {
        const separator = wsUrl.includes('?') ? '&' : '?';
        wsUrl = `${wsUrl}${separator}token=${encodeURIComponent(authToken)}`;
      }

      // Add CSRF token to URL for additional security
      const csrfToken = getCSRFToken();
      const csrfSeparator = wsUrl.includes('?') ? '&' : '?';
      wsUrl = `${wsUrl}${csrfSeparator}csrf=${encodeURIComponent(csrfToken)}`;

      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);

      // Set authentication timeout
      this.authTimer = setTimeout(() => {
        if (!this.isAuthenticated) {
          console.warn('[WebSocket] Authentication timeout - closing connection');
          this.disconnect();
        }
      }, this.config.authTimeout);
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);
      this.scheduleReconnect();
    }
  }

  disconnect(): void {
    this.clearTimers();
    this.reconnectAttempts = 0;
    this.isAuthenticated = false;

    if (this.ws) {
      this.ws.onclose = null; // Prevent reconnect on manual disconnect
      this.ws.close();
      this.ws = null;
    }

    this.setStatus('disconnected');
  }

  subscribe(channel: string, callback: MessageCallback): () => void {
    if (!this.subscribers.has(channel)) {
      this.subscribers.set(channel, new Set());
    }

    this.subscribers.get(channel)!.add(callback);

    // Send subscription message if connected and authenticated
    if (this.ws?.readyState === WebSocket.OPEN && this.isAuthenticated) {
      this.sendSubscribeMessage(channel);
    } else {
      this.pendingSubscriptions.add(channel);
    }

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscribers.get(channel);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscribers.delete(channel);
          this.sendUnsubscribeMessage(channel);
        }
      }
    };
  }

  subscribeToAll(callback: MessageCallback): () => void {
    this.globalSubscribers.add(callback);
    return () => {
      this.globalSubscribers.delete(callback);
    };
  }

  subscribeToStatus(callback: (status: ConnectionStatus) => void): () => void {
    this.statusSubscribers.add(callback);
    callback(this._status); // Immediately notify of current status
    return () => {
      this.statusSubscribers.delete(callback);
    };
  }

  send(message: Omit<WebSocketMessage, 'timestamp'>): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send message - not connected');
      return;
    }

    if (!this.isAuthenticated && message.channel !== 'system') {
      console.warn('[WebSocket] Cannot send message - not authenticated');
      return;
    }

    const fullMessage: WebSocketMessage = {
      ...message,
      timestamp: new Date().toISOString(),
    };

    this.ws.send(JSON.stringify(fullMessage));
  }

  private async handleOpen(): Promise<void> {
    console.log('[WebSocket] Connected, awaiting authentication...');
    this.setStatus('authenticating');

    // Send authentication message (in case URL token wasn't used)
    let authToken: string | null = null;
    if (this.authTokenProvider) {
      authToken = await this.authTokenProvider();
    }
    if (!authToken) {
      authToken = await secureStorage.getItem('access_token');
    }

    if (authToken) {
      this.send({
        channel: 'system',
        event: 'authenticate',
        data: { token: authToken },
      });
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      // Handle authentication response
      if (message.channel === 'system' && message.event === 'authenticated') {
        this.handleAuthenticated();
        return;
      }

      // Handle authentication error
      if (message.channel === 'system' && message.event === 'auth_error') {
        console.error('[WebSocket] Authentication failed:', message.data);
        this.disconnect();
        return;
      }

      // Handle heartbeat response
      if (message.event === 'pong') {
        return;
      }

      // Reject messages if not authenticated (except system messages)
      if (!this.isAuthenticated && message.channel !== 'system') {
        console.warn('[WebSocket] Received message while not authenticated');
        return;
      }

      // Notify global subscribers
      this.globalSubscribers.forEach((callback) => {
        try {
          callback(message);
        } catch (error) {
          console.error('[WebSocket] Global subscriber error:', error);
        }
      });

      // Notify channel subscribers
      const channelCallbacks = this.subscribers.get(message.channel);
      if (channelCallbacks) {
        channelCallbacks.forEach((callback) => {
          try {
            callback(message);
          } catch (error) {
            console.error('[WebSocket] Channel subscriber error:', error);
          }
        });
      }

      // Also notify wildcard subscribers (e.g., "alerts:*" matches "alerts:critical")
      this.subscribers.forEach((callbacks, pattern) => {
        if (pattern.endsWith(':*')) {
          const prefix = pattern.slice(0, -1);
          if (message.channel.startsWith(prefix)) {
            callbacks.forEach((callback) => {
              try {
                callback(message);
              } catch (error) {
                console.error('[WebSocket] Wildcard subscriber error:', error);
              }
            });
          }
        }
      });
    } catch (error) {
      console.error('[WebSocket] Message parse error:', error);
    }
  }

  private handleAuthenticated(): void {
    console.log('[WebSocket] Authenticated successfully');
    this.isAuthenticated = true;
    this.reconnectAttempts = 0;

    // Clear auth timeout
    if (this.authTimer) {
      clearTimeout(this.authTimer);
      this.authTimer = null;
    }

    this.setStatus('connected');

    // Subscribe to pending channels
    this.pendingSubscriptions.forEach((channel) => {
      this.sendSubscribeMessage(channel);
    });
    this.pendingSubscriptions.clear();

    // Re-subscribe to existing channels
    this.subscribers.forEach((_, channel) => {
      this.sendSubscribeMessage(channel);
    });

    // Start heartbeat
    this.startHeartbeat();
  }

  private handleClose(event: CloseEvent): void {
    console.log('[WebSocket] Disconnected:', event.code, event.reason);
    this.clearTimers();
    this.isAuthenticated = false;

    // Handle specific close codes
    if (event.code === 4001) {
      // Authentication required - don't reconnect without new token
      console.warn('[WebSocket] Authentication required');
      this.setStatus('disconnected');
      return;
    }

    if (event.code === 4003) {
      // Forbidden - user doesn't have access
      console.warn('[WebSocket] Access forbidden');
      this.setStatus('disconnected');
      return;
    }

    if (event.code !== 1000) {
      // Abnormal close - attempt reconnect
      this.scheduleReconnect();
    } else {
      this.setStatus('disconnected');
    }
  }

  private handleError(event: Event): void {
    console.error('[WebSocket] Error:', event);
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.log('[WebSocket] Max reconnect attempts reached');
      this.setStatus('disconnected');
      return;
    }

    this.reconnectAttempts++;
    this.setStatus('reconnecting');

    const delay = this.config.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1);
    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN && this.isAuthenticated) {
        this.send({
          channel: 'system',
          event: 'ping',
          data: null,
        });
      }
    }, this.config.heartbeatInterval);
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (this.authTimer) {
      clearTimeout(this.authTimer);
      this.authTimer = null;
    }
  }

  private sendSubscribeMessage(channel: string): void {
    this.send({
      channel: 'system',
      event: 'subscribe',
      data: { channel },
    });
  }

  private sendUnsubscribeMessage(channel: string): void {
    if (this.ws?.readyState === WebSocket.OPEN && this.isAuthenticated) {
      this.send({
        channel: 'system',
        event: 'unsubscribe',
        data: { channel },
      });
    }
  }
}

// Create singleton instance
const WS_URL = import.meta.env.VITE_WS_URL || `ws://${window.location.host}/ws`;
export const websocketService = new WebSocketService({ url: WS_URL });

export default websocketService;
