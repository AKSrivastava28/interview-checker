class IntegrityWebSocketClient {
    constructor(roomId, role) {
        this.roomId = roomId;
        this.role = role; // "candidate" or "dashboard"
        this.socket = null;
        this.listeners = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.isConnecting = false;
    }

    connect() {
        if (this.isConnecting || (this.socket && this.socket.readyState === WebSocket.OPEN)) {
            return;
        }
        this.isConnecting = true;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        
        let wsUrl;
        if (this.role === "candidate") {
            wsUrl = `${protocol}//${host}/ws/${this.roomId}`;
        } else {
            wsUrl = `${protocol}//${host}/ws/dashboard/${this.roomId}`;
        }

        console.log(`[WebSocket] Connecting to ${wsUrl}...`);
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log(`[WebSocket] Connected successfully as ${this.role}`);
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this.emit('open');
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.emit('message', data);
                if (data.type) {
                    this.emit(data.type, data);
                }
            } catch (err) {
                console.error("[WebSocket] Failed to parse JSON message:", err);
            }
        };

        this.socket.onclose = (event) => {
            console.warn(`[WebSocket] Closed (code: ${event.code})`);
            this.isConnecting = false;
            this.emit('close', event);

            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
                console.log(`[WebSocket] Reconnecting in ${delay}ms...`);
                setTimeout(() => this.connect(), delay);
            }
        };

        this.socket.onerror = (err) => {
            console.error("[WebSocket] Error:", err);
            this.isConnecting = false;
            this.emit('error', err);
        };
    }

    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        } else {
            console.warn("[WebSocket] Cannot send message, socket not open.");
        }
    }

    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(cb => cb(data));
        }
    }
}

window.IntegrityWebSocketClient = IntegrityWebSocketClient;
