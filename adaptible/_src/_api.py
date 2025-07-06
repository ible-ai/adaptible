"""Standard Interaction Logic for a Stateful LLM"""

import asyncio
from asyncio import log
import collections
import os
import time
import threading
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import tqdm
from vizible import green

from ._classes import (
    InteractionHistory,
    InteractionRequest,
    InteractionResponse,
    ReviewResponse,
    SyncResponse,
)
from ._llm import StatefulLLM

App = FastAPI(
    title="Stateful Self-Improving LLM Server",
    description="An API for a stateful LLM that tries to improve itself over time.",
)
App.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# In-memory store or interaction history from the current session.
interaction_history: List[InteractionHistory] = []
unreviewed_interaction_history_indices: List[int] = []

outstanding_tasks: collections.deque[Any] = collections.deque([])

# Instantiate the stateful model. This is a singleton for the lifecycle of the App.
model = StatefulLLM()


@App.post("/interact", response_model=InteractionResponse)
async def interact_with_model(request: InteractionRequest):
    """
    Main endpoint for interacting with the LLM (Forward-pass).
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    print(f"Received request: {request}")
    # Generate the response using the current state of the model.
    response_text = model.generate_response(request.prompt)

    # Store the interaction for later review
    interaction_idx = len(interaction_history)
    unreviewed_interaction_history_indices.append(len(interaction_history))
    interaction_history.append(
        InteractionHistory(
            idx=interaction_idx,
            user_input=request.prompt,
            llm_response=response_text,
            reviewed=False,
            timestamp=time.time(),
        )
    )
    return {"response": response_text, "interaction_idx": interaction_idx}


@App.post("/trigger_review", response_model=ReviewResponse)
async def trigger_review_cycle():
    """Manually triggers the self-correction and training cycle."""
    unreviewed_count = len(unreviewed_interaction_history_indices)
    if unreviewed_count == 0:
        return {
            "message": "No unreviewed interactions to process.",
            "unreviewed_count": unreviewed_count,
        }
    unreviewed_interaction_history = [
        interaction_history[idx] for idx in unreviewed_interaction_history_indices
    ]
    outstanding_tasks.append(
        asyncio.to_thread(model.self_correct_and_train, unreviewed_interaction_history)
    )
    return {
        "message": "Self-correction and training cycle has been initiated in the background.",
        "unreviewed_count": unreviewed_count,
    }


@App.get("/sync", response_model=SyncResponse)
async def sync_server_for_background_tasks():
    """Waits for any tasks to background complete."""
    start_time = time.time()
    num_tasks = len(outstanding_tasks)
    print("Waiting for model state to stabilize...")
    lock = threading.Lock()
    with (
        lock,
        tqdm.tqdm(desc="Waiting for server to sync.", unit=" Seconds") as server_pbar,
    ):
        green("Finished background tasks")
        while not model.ok:
            log.logger.info(
                "Waiting for model server to sync. Is model is %s ok.",
                "" if model.ok else "not ",
            )
            server_pbar.update(1)
            await asyncio.sleep(1)
    green("Model has reached a stable state!!!")
    elapsed_time = time.time() - start_time
    return {
        "message": "Sync'd all background tasks.",
        "tasks_count": num_tasks,
        "elapsed_time": elapsed_time,
    }


@App.get("/history")
async def get_history():
    """Returns the full interaction history."""
    return {"history": interaction_history}


@App.get("/status")
async def check_is_running():
    """Basic response to signal that the server is operational."""
    return {"status": "up"}
