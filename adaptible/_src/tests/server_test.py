"""Tests for the MutableHostedLLM server in _server.py."""

import asyncio
import unittest
from unittest import mock

from fastapi import FastAPI
import uvicorn

import adaptible


def create_stub_app() -> FastAPI:
    """Create a minimal FastAPI app for testing server lifecycle."""
    app = FastAPI()

    @app.get("/status")
    async def status():
        return {"status": "up"}

    return app


class MutableHostedLLMInitTest(unittest.TestCase):
    """Tests for MutableHostedLLM initialization."""

    def test_init_default_values(self):
        """Should initialize with default host and port."""
        server = adaptible.MutableHostedLLM(app=create_stub_app())
        self.assertEqual(server.config.host, "127.0.0.1")
        self.assertEqual(server.config.port, 8000)

    def test_init_creates_startup_event(self):
        """Should create startup event for coordination."""
        server = adaptible.MutableHostedLLM(app=create_stub_app())
        self.assertIsInstance(server._startup_done, asyncio.Event)

    def test_init_serve_task_is_none(self):
        """Serve task should be None before starting."""
        server = adaptible.MutableHostedLLM(app=create_stub_app())
        self.assertIsNone(server._serve_task)


class MutableHostedLLMCustomPortTest(unittest.TestCase):
    """Tests for MutableHostedLLM with custom settings."""

    def test_init_custom_host_port(self):
        """Should accept custom host and port."""
        server = adaptible.MutableHostedLLM(
            host="0.0.0.0", port=9000, app=create_stub_app()
        )

        self.assertEqual(server.config.host, "0.0.0.0")
        self.assertEqual(server.config.port, 9000)


class MutableHostedLLMStartupTest(unittest.TestCase):
    """Tests for MutableHostedLLM startup method."""

    def test_startup_sets_event(self):
        """Startup should set the startup_done event."""
        server = adaptible.MutableHostedLLM(app=create_stub_app())

        with mock.patch.object(uvicorn.Server, "startup", new_callable=mock.AsyncMock):
            asyncio.run(server.startup())

        self.assertTrue(server._startup_done.is_set())


class MutableHostedLLMUpDownTest(unittest.TestCase):
    """Tests for MutableHostedLLM up/down methods."""

    def test_up_creates_serve_task(self):
        """Up should create a serve task."""
        server = adaptible.MutableHostedLLM(app=create_stub_app())

        async def mock_serve(sockets=None):
            await asyncio.sleep(0.1)

        server.serve = mock_serve  # type: ignore

        async def test():
            server._startup_done.set()
            await server.up()
            self.assertIsNotNone(server._serve_task)
            server.should_exit = True

        asyncio.run(test())

    def test_down_sets_should_exit(self):
        """Down should set should_exit flag."""
        server = adaptible.MutableHostedLLM(app=create_stub_app())

        async def dummy_task():
            pass

        async def test():
            server._serve_task = asyncio.create_task(dummy_task())
            await server.down()
            self.assertTrue(server.should_exit)

        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()
