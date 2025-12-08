"""Server definition for spinning up a StatefulLLM instance."""

import asyncio
import logging
import socket
import sys

import uvicorn
from fastapi import FastAPI

from .._api import Adaptible

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)  # Get a logger for this module


class MutableHostedLLM(uvicorn.Server):
    """Host server to interact with a stateful LLM"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        app: FastAPI | None = None,
    ):
        """Initialize the server.

        Args:
            host: Host address to bind to.
            port: Port to listen on.
            app: Optional FastAPI app. If not provided, creates Adaptible with default model.
        """
        if app is None:
            app = Adaptible().app
        super().__init__(config=uvicorn.Config(app, host=host, port=port))
        self._startup_done = asyncio.Event()
        self._serve_task = None
        self.should_exit = False

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        """Override uvicorn startup"""
        await super().startup(sockets=sockets)
        self.config.setup_event_loop()
        self._startup_done.set()

    async def up(self) -> None:
        """Start up server asynchronously"""
        self._serve_task = asyncio.create_task(self.serve())
        await self._startup_done.wait()

    async def down(self) -> None:
        """Shut down server asynchronously"""
        self.should_exit = True
        assert self._serve_task is not None
        await self._serve_task
