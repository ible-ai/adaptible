"""Server definition for spinning up a StatefulLLM instance."""

import asyncio
import logging
import socket
import sys

from fastapi import FastAPI
import uvicorn

from ._api import app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__) # Get a logger for this module


class MutableHostedLLM(uvicorn.Server):
    """Host server to interact with a stateful LLM"""

    def __init__(self, uvicorn_app: FastAPI = app, host: str = '127.0.0.1', port: int = 8000):
        super().__init__(config=uvicorn.Config(uvicorn_app, host=host, port=port))
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
        await self._serve_task
