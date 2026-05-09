import { useEffect, useState } from 'react';

export interface SwimmerAnalyticsData {
  swimmer_count: number;
  crowd_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  maintenance_urgency: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  fps: number;
  status: {
    swimmer_count: number;
    occupancy_capacity: number;
    occupancy_percentage: number;
    crowd_density_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    real_time_status: string;
    maintenance_recommendations: Array<{ action: string; priority: string; overdue_by: number }>;
  };
}

interface UseSwimmerAnalyticsState {
  data: SwimmerAnalyticsData | null;
  loading: boolean;
  error: string | null;
  connected: boolean;
}

export function useSwimmerAnalytics(clientId: string = 'dashboard'): UseSwimmerAnalyticsState {
  const [state, setState] = useState<UseSwimmerAnalyticsState>({
    data: null,
    loading: true,
    error: null,
    connected: false,
  });

  useEffect(() => {
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1';
    const wsBaseUrl = apiBaseUrl.replace(/^http/, 'ws');
    const wsUrl = `${wsBaseUrl}/ws/swimmer-analytics?client_id=${encodeURIComponent(clientId)}`;

    let socket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let isManualClose = false;

    const connect = () => {
      try {
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
          setState((prev) => ({
            ...prev,
            connected: true,
            loading: false,
            error: null,
          }));
        };

        socket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'swimmer.analytics') {
              setState((prev) => ({
                ...prev,
                data: message as SwimmerAnalyticsData,
                error: null,
              }));
            }
          } catch {
            // Silently ignore parse errors
          }
        };

        socket.onerror = () => {
          setState((prev) => ({
            ...prev,
            error: 'Connection error',
            connected: false,
          }));
        };

        socket.onclose = () => {
          setState((prev) => ({
            ...prev,
            connected: false,
          }));

          // Attempt reconnect if not manually closed
          if (!isManualClose && !reconnectTimeout) {
            reconnectTimeout = setTimeout(() => {
              reconnectTimeout = null;
              connect();
            }, 3000);
          }
        };
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: 'Failed to connect',
          connected: false,
        }));
      }
    };

    connect();

    return () => {
      isManualClose = true;
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (socket) {
        socket.close();
      }
    };
  }, [clientId]);

  return state;
}
