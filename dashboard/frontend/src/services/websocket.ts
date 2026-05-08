type WebSocketMessageHandler = (data: unknown) => void;

export class DashboardWebSocketService {
  private socket: WebSocket | null = null;

  connect(url: string, onMessage: WebSocketMessageHandler) {
    this.socket = new WebSocket(url);

    this.socket.onmessage = (event) => {
      onMessage(JSON.parse(event.data));
    };
  }

  disconnect() {
    this.socket?.close();
    this.socket = null;
  }
}

export const dashboardWebSocketService = new DashboardWebSocketService();
