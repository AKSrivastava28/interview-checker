import logging
from typing import Dict, List
from fastapi import WebSocket

logger = logging.getLogger("websocket_manager")

class ConnectionManager:
    def __init__(self):
        # session_id -> List of candidate WebSockets
        self.candidate_sockets: Dict[str, List[WebSocket]] = {}
        # session_id -> List of dashboard WebSockets
        self.dashboard_sockets: Dict[str, List[WebSocket]] = {}

    async def connect_candidate(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.candidate_sockets:
            self.candidate_sockets[session_id] = []
        self.candidate_sockets[session_id].append(websocket)
        logger.info(f"Candidate WebSocket connected to session {session_id}")

    def disconnect_candidate(self, session_id: str, websocket: WebSocket):
        if session_id in self.candidate_sockets:
            if websocket in self.candidate_sockets[session_id]:
                self.candidate_sockets[session_id].remove(websocket)

    async def connect_dashboard(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.dashboard_sockets:
            self.dashboard_sockets[session_id] = []
        self.dashboard_sockets[session_id].append(websocket)
        logger.info(f"Dashboard WebSocket connected to session {session_id}")

    def disconnect_dashboard(self, session_id: str, websocket: WebSocket):
        if session_id in self.dashboard_sockets:
            if websocket in self.dashboard_sockets[session_id]:
                self.dashboard_sockets[session_id].remove(websocket)

    async def send_to_dashboard(self, session_id: str, message: dict):
        if session_id in self.dashboard_sockets:
            to_remove = []
            for ws in self.dashboard_sockets[session_id]:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.warning(f"Error sending message to dashboard in session {session_id}: {e}")
                    to_remove.append(ws)
            for ws in to_remove:
                self.disconnect_dashboard(session_id, ws)

    async def send_to_candidate(self, session_id: str, message: dict):
        if session_id in self.candidate_sockets:
            to_remove = []
            for ws in self.candidate_sockets[session_id]:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.warning(f"Error sending message to candidate in session {session_id}: {e}")
                    to_remove.append(ws)
            for ws in to_remove:
                self.disconnect_candidate(session_id, ws)

ws_manager = ConnectionManager()
