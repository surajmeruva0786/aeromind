"""WebSocket server pushing `PredictionEvent`s to downstream dashboards
(README §13), e.g. the Streamlit app in Phase 10. A minimal one-way
telemetry broadcast built on the `websockets` library — no request/response
protocol, inbound client messages are ignored.
"""

from __future__ import annotations

import asyncio
import json

import websockets
from websockets.asyncio.server import Server, ServerConnection

from src.inference.stream import PredictionEvent
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def event_to_json(event: PredictionEvent) -> str:
    return json.dumps(
        {
            "timestamp": event.timestamp,
            "workload_probs": event.workload_probs.tolist(),
            "fatigue_probs": event.fatigue_probs.tolist(),
            "smoothed_workload_probs": event.smoothed_workload_probs.tolist(),
            "smoothed_fatigue_probs": event.smoothed_fatigue_probs.tolist(),
        }
    )


class PredictionBroadcastServer:
    """Any connected client receives every `PredictionEvent` pushed via
    `broadcast()`. `start()`/`stop()` are async — call from an asyncio
    event loop (see `src/inference/README.md` for a usage example)."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set[ServerConnection] = set()
        self._server: Server | None = None

    async def _handler(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for _ in websocket:
                pass  # push-only feed; inbound messages are ignored
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected (%d total)", len(self._clients))

    async def broadcast(self, event: PredictionEvent) -> None:
        if not self._clients:
            return
        payload = event_to_json(event)
        results = await asyncio.gather(
            *(client.send(payload) for client in list(self._clients)), return_exceptions=True
        )
        for client, result in zip(list(self._clients), results):
            if isinstance(result, Exception):
                self._clients.discard(client)

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, self.host, self.port)
        logger.info("Prediction WebSocket server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
