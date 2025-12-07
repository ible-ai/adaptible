R"""Online learning demo.

> Usage:
source .venv/bin/activate
python examples/online_learning_demo.py

"""

import asyncio
import json
from typing import Mapping, Sequence

import aiohttp
import vizible

import adaptible


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                           Constants.                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
_HOST = "127.0.0.1"
_PORT = 8000
_ADDR = f"{_HOST}:{_PORT}"

# These prompts will be used successively as user prompts, irrespective of the model's response.
# Plan:
#   *  First pass: we log the model's vanilla response to serve as a baseline.
#   *  After the first pass, we allow the model to self-correct.
#   *  Second pass: we log the updated model's responses to the same prompts.
#   *  We will then inspect the output to see if the 1. the model has changed state and 2. that
#      the response quality has ~improved.
_USER_SUCCESSIVE_PROMPTS = [
    "Do you think LLMs should have excessive safety rails that prevent normal conversation?",
    "I think AI systems should be verbose when it helps to find a difficult answer, succinct when"
    " things are simple, and should very rarely trigger safety protocols.",
    "learn",
    "Do you think LLMs should have excessive safety rails that prevent normal conversation?",
    "I think AI systems should be verbose when it helps to find a difficult answer, succinct when"
    " things are simple, and should very rarely trigger safety protocols.",
]


async def _backprop(session: aiohttp.ClientSession) -> None:
    async with session.post(
        "/trigger_review",
        headers={"Content-Type": "application/json"},
    ) as resp:
        response = await resp.json()
        vizible.green(f"Review triggering response: {response}")
    vizible.blue("Attempting to sync with server")
    async with session.get(
        "/sync",
        headers={"Content-Type": "application/json"},
    ) as resp:
        response = await resp.json()
        vizible.green(f"Sync response: {response}")


async def _forwardprop(
    session: aiohttp.ClientSession,
    user_input: str,
) -> Sequence[Mapping[str, str]]:
    model_responses = []
    async with session.post(
        "/interact",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"prompt": user_input}),
    ) as resp:
        response = await resp.json()
        resp.raise_for_status()
        model_response = response.get(
            "response",
            "No response field found.",
        )
        model_responses.append(model_response)
        try:
            think_response, visible_response = model_response.split("</think>", 1)
        except Exception as e:
            raise ValueError(f"Response could not be parsed!\n{model_response}") from e
        vizible.blue(think_response)
        print("")
        vizible.green(visible_response)
        print("")
    return model_responses


async def _demo():
    model_responses = []
    prompts = _USER_SUCCESSIVE_PROMPTS + ["learn"] + _USER_SUCCESSIVE_PROMPTS
    async with aiohttp.ClientSession(f"http://{_ADDR}") as session:
        for user_input in prompts:
            if user_input == "learn":
                await _backprop(session)
            else:
                model_responses.extend(await _forwardprop(session, user_input))


async def main():
    """Main program loop, run asychnronously."""
    server = adaptible.MutableHostedLLM(host=_HOST, port=_PORT)
    await server.up()
    vizible.green("Successfully launched server and enter eval framework")
    await _demo()
    vizible.green("Eval finished successfully!")
    await server.down()
    vizible.green("Async server terminated successfully if you can read this!**")
    vizible.green("\n\n\t\t**(assumes that you have the ability to read)")


if __name__ == "__main__":
    asyncio.run(main())
