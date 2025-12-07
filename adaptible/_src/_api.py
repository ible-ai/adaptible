"""Standard Interaction Logic for a Stateful LLM"""

import asyncio
from asyncio import log
import collections
import os
import time
import threading
from typing import Any, List, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import tqdm
import vizible

from ._classes import (
    InteractionHistory,
    InteractionRequest,
    InteractionResponse,
    ReviewResponse,
    SyncResponse,
)
from ._llm import StatefulLLM


class ModelProtocol(Protocol):
    """Protocol defining the model interface for Adaptible."""

    ok: bool

    def generate_response(self, prompt: str) -> str: ...
    def stream_response(self, prompt: str) -> Any: ...
    def self_correct_and_train(
        self,
        interaction_history: List[InteractionHistory],
        indices_to_review: List[int] | None = None,
        verbose: bool = False,
    ) -> bool: ...


class Adaptible:
    """Main API app for the stateful LLM."""

    def __init__(self, app: FastAPI | None = None, model: ModelProtocol | None = None):
        """Initialize the Adaptible API with endpoints.

        Args:
            app: Optional FastAPI app to use. Creates a new one if not provided.
            model: Optional model conforming to ModelProtocol. Creates a StatefulLLM if not provided.
        """
        if app is not None:
            self.app = app
        else:
            self.app = FastAPI(
                title="Stateful Self-Improving LLM Server",
                description="An API for a stateful LLM that tries to improve itself over time.",
            )
        self.app.mount(
            "/static",
            StaticFiles(
                directory=os.path.join(os.path.dirname(__file__), "static"),
                html=True,
            ),
            name="static",
        )

        # In-memory store or interaction history from the current session.
        self.interaction_history: List[InteractionHistory] = []
        self.unreviewed_interaction_history_indices: List[int] = []

        self.outstanding_tasks: collections.deque[Any] = collections.deque([])

        # Use provided model or instantiate a new one.
        self.model = model if model is not None else StatefulLLM()

        @self.app.post("/interact", response_model=InteractionResponse)
        async def interact_with_model(request: InteractionRequest):
            """
            Main endpoint for interacting with the LLM (Forward-pass).
            """
            if not request.prompt:
                raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
            print(f"Received request: {request}")
            # Generate the response using the current state of the model.
            response_text = self.model.generate_response(request.prompt)

            # Store the interaction for later review
            interaction_idx = len(self.interaction_history)
            self.unreviewed_interaction_history_indices.append(
                len(self.interaction_history)
            )
            self.interaction_history.append(
                InteractionHistory(
                    idx=interaction_idx,
                    user_input=request.prompt,
                    llm_response=response_text,
                    reviewed=False,
                    timestamp=time.time(),
                )
            )
            return {"response": response_text, "interaction_idx": interaction_idx}

        @self.app.post("/stream_interact")
        async def stream_interact_with_model(request: InteractionRequest):
            """Stream interaction with model."""
            # TODO - enable logging.
            return StreamingResponse(
                self.model.stream_response(request.prompt), media_type="text/plain"
            )

        @self.app.post("/trigger_review", response_model=ReviewResponse)
        async def trigger_review_cycle():
            """Manually triggers the self-correction and training cycle."""
            unreviewed_count = len(self.unreviewed_interaction_history_indices)
            if unreviewed_count == 0:
                return {
                    "message": "No unreviewed interactions to process.",
                    "unreviewed_count": unreviewed_count,
                }
            unreviewed_interaction_history = [
                self.interaction_history[idx]
                for idx in self.unreviewed_interaction_history_indices
            ]
            self.outstanding_tasks.append(
                asyncio.to_thread(
                    self.model.self_correct_and_train, unreviewed_interaction_history
                )
            )
            return {
                "message": "Self-correction and training cycle has been initiated in the background.",
                "unreviewed_count": unreviewed_count,
            }

        @self.app.get("/sync", response_model=SyncResponse)
        async def sync_server_for_background_tasks():
            """Waits for any tasks to background complete."""
            start_time = time.time()
            num_tasks = len(self.outstanding_tasks)
            print("Waiting for model state to stabilize...")
            lock = threading.Lock()
            with (
                lock,
                tqdm.tqdm(
                    desc="Waiting for server to sync.", unit=" Seconds"
                ) as server_pbar,
            ):
                vizible.green("Finished background tasks")
                while not self.model.ok:
                    log.logger.info(
                        "Waiting for model server to sync. Is model is %s ok.",
                        "" if self.model.ok else "not ",
                    )
                    server_pbar.update(1)
                    await asyncio.sleep(1)
            vizible.green("Model has reached a stable state!!!")
            elapsed_time = time.time() - start_time
            return {
                "message": "Sync'd all background tasks.",
                "tasks_count": num_tasks,
                "elapsed_time": elapsed_time,
            }

        @self.app.get("/history")
        async def get_history():
            """Returns the full interaction history."""
            return {"history": self.interaction_history}

        @self.app.get("/status")
        async def check_is_running():
            """Basic response to signal that the server is operational."""
            return {"status": "up"}
