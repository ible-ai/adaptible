"""Client-facing API: preserve runtime chat payloads and record complete turns."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .thinking import completion_details, generation_mode, thinking_complete

logger = logging.getLogger(__name__)

_REPAIRABLE = {"new", "pending", "kept", "rejected", "unchanged", "skipped", "failed"}

# Generation stopped because it hit its token budget, as opposed to a stream
# that was cut off before the runtime said why.
_OUT_OF_TOKENS = ("length", "max_tokens")


def _ran_out_of_tokens(status, details):
    """Whether this turn is a complete generation that simply hit its budget.

    A turn the model ran away on is still a turn the user can call wrong, and
    the flagged text is never training input -- repair regenerates the
    correction from the reference. Refusing feedback on it meant an item the
    model answered badly enough to exhaust its budget could never be repaired,
    while cycles_mlx.py scores that generation as a miss and goes on to sample
    and train. A stream that died mid-delivery is different: nothing says what
    the model would have produced, so it stays unratable.
    """
    if status != "incomplete":
        return False
    try:
        recorded = json.loads(details or "{}")
    except ValueError:
        return False
    # An answer the model ran away on can be rated; nothing but an unfinished
    # thought cannot, because no answer was produced to call wrong.
    return bool(
        recorded.get("finish_reason") in _OUT_OF_TOKENS
        and (recorded.get("content") or "").strip()
    )


def create_app(controller):
    runtime, store = controller.runtime, controller.store
    sessions = {}

    @asynccontextmanager
    async def lifespan(app):
        try:
            await controller.start()
            # Only now is the model actually being served. A runtime the
            # wrapper hosts itself (llama.cpp, vLLM) launches its server in
            # start(), so probing before this point cannot connect and the
            # probe's fail-safe silently leaves the model marked as not
            # reasoning -- which routes a reasoning model down the plain path
            # for the whole session. Ollama and LM Studio talk to a server that
            # is already up, so they were unaffected and the split went
            # unnoticed.
            if hasattr(runtime, "detect_reasoning"):
                await runtime.detect_reasoning()
            yield
        finally:
            await controller.close()

    app = FastAPI(title="Adaptible runtime wrapper", lifespan=lifespan)

    async def read_body(request):
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Request body must be JSON.") from exc
        if not isinstance(body, dict):
            raise HTTPException(400, "Request body must be a JSON object.")
        return body

    def check_model(body):
        model = body.get("model", runtime.name)
        if not isinstance(model, str) or model not in {
            runtime.name,
            runtime.name + ":latest",
        }:
            raise HTTPException(404, f"This wrapper serves {runtime.name!r}.")

    def turn_mode(mode, details):
        """What this turn actually was, not what the model usually does.

        ``always_reasons`` is inferred from a single startup probe, but a model
        can reason on some prompts and answer others directly -- the same
        DeepSeek-R1 distill returns a thought for "reply with the word ready"
        and none for "what is the capital of Turkey?". Demanding a thought on
        every turn records each direct answer as incomplete, which makes it
        unflaggable, which leaves the model uncorrectable on every runtime.

        A thinking turn the caller explicitly asked for is different: if that
        control is not honoured the turn really is unusable, so it is left
        alone.
        """
        if not mode.get("thinking") or details["reasoning"].strip():
            return mode
        source = list(mode.get("source") or [])
        if "model_reasons_unconditionally" not in source:
            return mode
        return {
            **mode,
            "thinking": False,
            "source": source + ["turn_without_reasoning"],
        }

    def finish(idx, details, complete, mode=None):
        with store.db:
            if mode is not None:
                store.db.execute(
                    "UPDATE interactions SET generation_mode=? WHERE id=?",
                    (json.dumps(mode), idx),
                )
            store.db.execute(
                "UPDATE interactions SET response=?, response_details=?, status=? WHERE id=?",
                (
                    details["content"],
                    json.dumps(details),
                    "new" if complete else "incomplete",
                    idx,
                ),
            )

    async def proxy(request, path, body, messages, *, plain=False, session=None):
        if controller.failed:
            raise HTTPException(503, controller.failed)
        controller.touch()
        await controller.gate.acquire()
        handed_off = False
        upstream = None
        try:
            if controller.failed:
                raise HTTPException(503, controller.failed)
            if session:
                messages = list(sessions.get(session, [])) + [messages[-1]]
                body = {**body, "messages": messages}
            # The active model can have changed while this request waited.
            mode = generation_mode(
                body,
                getattr(runtime, "architecture", None),
                native=runtime.native,
                path=path,
                provider=type(runtime).__name__,
                always_reasons=getattr(runtime, "always_reasons", False),
            )
            adaptation = await controller.route(messages, mode, body=body)
            frozen = adaptation["adapter"] == "base"
            if hasattr(runtime, "prepare_payload"):
                payload = await runtime.prepare_payload(body, frozen=frozen)
            else:
                payload = runtime.payload(body, frozen=frozen)
            if hasattr(runtime, "normalize_payload"):
                payload = runtime.normalize_payload(payload, path=path)
            request_options = {}
            if hasattr(runtime, "generation_timeout"):
                request_options["timeout"] = runtime.generation_timeout(
                    mode.get("thinking", False)
                )
            upstream = await runtime.client.send(
                runtime.client.build_request(
                    "POST", runtime.url + path, json=payload, **request_options
                ),
                stream=True,
            )
            if upstream.is_error:
                data = await upstream.aread()
                return JSONResponse(
                    {"error": data.decode(errors="replace")},
                    status_code=upstream.status_code,
                )
            idx = store.record(messages, "", generation_mode=mode)
            store.outcome(idx, "streaming", "")
            headers = {"X-Interaction-Idx": str(idx)}
            headers["X-Adaptible-Adapter"] = adaptation["adapter"]
            if adaptation["scope"] is not None:
                headers["X-Adaptible-Scope"] = str(adaptation["scope"])
            if session:
                headers["Set-Cookie"] = (
                    f"adaptible-session={session}; HttpOnly; SameSite=Strict; Path=/"
                )
            if not body.get("stream", path.startswith("/api/")):
                data = json.loads(await upstream.aread())
                native = path.startswith("/api/")
                message = (
                    data.get("message", data)
                    if native
                    else data["choices"][0]["message"]
                )
                stop = (
                    data.get("done_reason")
                    if native
                    else data["choices"][0].get("finish_reason")
                )
                details = completion_details(message, stop)
                details["adaptation"] = adaptation
                text = details["content"]
                actual = turn_mode(mode, details)
                completed = (
                    stop not in ("length", "max_tokens")
                    and details["framing_valid"]
                    and (not actual.get("thinking") or thinking_complete(details))
                )
                finish(idx, details, completed, actual)
                data["model"] = runtime.name
                if completed and session:
                    sessions[session] = messages + [
                        dict(role="assistant", content=text)
                    ]
                if plain:
                    prefix = details["reasoning_prefix"] or (
                        "<think>\n" + details["reasoning"]
                    )
                    rendered = (prefix + text) if details["reasoning"] else text
                    data = {"response": rendered, "interaction_idx": idx}
                return JSONResponse(data, headers=headers)

            async def relay():
                pieces = []
                thoughts = []
                stop = None
                thought_open = False
                complete = False
                try:
                    async for line in upstream.aiter_lines():
                        if not line:
                            continue
                        if line == "data: [DONE]":
                            complete = True
                            if not plain:
                                yield "data: [DONE]\n\n"
                            break
                        native = path.startswith("/api/")
                        if not native and not line.startswith("data: "):
                            continue
                        data = json.loads(line if native else line[6:])
                        if data.get("error"):
                            raise RuntimeError(str(data["error"]))
                        data["model"] = runtime.name
                        if native:
                            delta = data.get("message", data)
                            text = data.get(
                                "response", data.get("message", {}).get("content", "")
                            )
                            complete = data.get("done") is True
                            stop = data.get("done_reason") or stop
                        else:
                            choices = data.get("choices") or []
                            delta = choices[0].get("delta", {}) if choices else {}
                            stop = (
                                choices[0].get("finish_reason") if choices else None
                            ) or stop
                            text = (
                                (choices[0].get("delta", {}).get("content") or "")
                                if choices
                                else ""
                            )
                        pieces.append(text)
                        reasoning = next(
                            (
                                delta[key]
                                for key in (
                                    "reasoning_content",
                                    "reasoning",
                                    "thinking",
                                )
                                if isinstance(delta.get(key), str) and delta[key]
                            ),
                            "",
                        )
                        thoughts.append(reasoning)
                        if plain:
                            if reasoning:
                                if not thought_open:
                                    yield "<think>\n"
                                    thought_open = True
                                yield reasoning
                            if text and thought_open:
                                yield "\n</think>\n\n"
                                thought_open = False
                            yield text
                        else:
                            encoded = json.dumps(data)
                            yield (
                                encoded + "\n"
                                if native
                                else "data: " + encoded + "\n\n"
                            )
                        if complete:
                            break
                finally:
                    # Each cleanup step is isolated: a failure here must not
                    # replace an exception already propagating from the stream,
                    # and must not skip releasing the gate.
                    try:
                        details = completion_details(
                            dict(content="".join(pieces), reasoning="".join(thoughts)),
                            stop,
                        )
                        details["adaptation"] = adaptation
                        text = details["content"]
                        actual = turn_mode(mode, details)
                        complete = (
                            complete
                            and stop not in ("length", "max_tokens")
                            and details["framing_valid"]
                            and (
                                not actual.get("thinking") or thinking_complete(details)
                            )
                        )
                        finish(idx, details, complete, actual)
                        if complete and session:
                            sessions[session] = messages + [
                                dict(role="assistant", content=text)
                            ]
                    except Exception:
                        logger.exception(
                            "Could not record the completed turn for interaction %s.",
                            idx,
                        )
                    try:
                        await upstream.aclose()
                    except Exception:
                        logger.exception("Could not close the upstream response.")
                    controller.touch()
                    controller.gate.release()

            handed_off = True
            media = (
                "text/plain"
                if plain
                else (
                    "application/x-ndjson"
                    if path.startswith("/api/")
                    else "text/event-stream"
                )
            )
            return StreamingResponse(relay(), headers=headers, media_type=media)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Runtime request failed: {exc}") from exc
        finally:
            if not handed_off:
                try:
                    if upstream:
                        await upstream.aclose()
                finally:
                    controller.touch()
                    controller.gate.release()

    async def chat(request, path):
        body = await read_body(request)
        check_model(body)
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(m, dict) for m in messages)
        ):
            raise HTTPException(400, "messages must be a nonempty list.")
        # For the OpenAI endpoint, omission of stream means false.
        body.setdefault("stream", path.startswith("/api/"))
        return await proxy(request, path, body, messages)

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        return await chat(request, "/v1/chat/completions")

    @app.post("/api/chat")
    async def native_chat(request: Request):
        if not runtime.native:
            raise HTTPException(404, "Use /v1/chat/completions for this runtime.")
        return await chat(request, "/api/chat")

    @app.post("/api/generate")
    async def native_generate(request: Request):
        if not runtime.native:
            raise HTTPException(404, "Use /v1/chat/completions for this runtime.")
        body = await read_body(request)
        check_model(body)
        if not isinstance(body.get("prompt"), str) or not body["prompt"]:
            raise HTTPException(400, "A text prompt is required.")
        messages = (
            [dict(role="system", content=body["system"])] if body.get("system") else []
        )
        messages.append(dict(role="user", content=body["prompt"]))
        return await proxy(request, "/api/generate", body, messages)

    @app.post("/interact")
    @app.post("/stream_interact")
    async def cli_chat(request: Request):
        body = await read_body(request)
        if not isinstance(body.get("prompt"), str) or not body["prompt"].strip():
            raise HTTPException(400, "A text prompt is required.")
        session = request.cookies.get("adaptible-session") or uuid.uuid4().hex
        messages = (
            list(sessions.get(session, [])) if body.get("use_history", True) else []
        )
        messages.append(dict(role="user", content=body["prompt"]))
        payload = dict(
            messages=messages,
            stream=request.url.path == "/stream_interact",
            temperature=0,
            max_tokens=runtime.max_tokens,
        )
        if getattr(runtime, "architecture", None) == "qwen3":
            # The terminal exercises real Qwen3 reasoning; review preserves it.
            # Compatible public API routes preserve their callers' settings.
            payload["reasoning_effort"] = "medium"
            if not runtime.native:
                payload["chat_template_kwargs"] = {"enable_thinking": True}
        return await proxy(
            request,
            "/v1/chat/completions",
            payload,
            messages,
            plain=True,
            session=session if body.get("use_history", True) else None,
        )

    @app.post("/new_chat")
    async def new_chat(request: Request):
        async with controller.gate:
            sessions.pop(request.cookies.get("adaptible-session"), None)
        return {"message": "New conversation started."}

    @app.post("/feedback")
    async def feedback(request: Request):
        body = await read_body(request)
        idx, thumbs = body.get("interaction_idx"), body.get("thumbs", "down")
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or thumbs not in ("up", "down")
        ):
            raise HTTPException(400, "Provide interaction_idx and thumbs: up or down.")
        row = store.db.execute(
            "SELECT status, response_details FROM interactions WHERE id=?", (idx,)
        ).fetchone()
        if row and row[0] not in _REPAIRABLE and not _ran_out_of_tokens(*row):
            raise HTTPException(
                409, "Wait for the response/review to finish before rating it."
            )
        note = body.get("note", "")
        if not isinstance(note, str):
            raise HTTPException(400, "note must be text.")
        # Phrasings this correction should be judged against. The repair loop
        # otherwise generates its own, which is not the same set the caller
        # cares about -- and for a reproduction, the experiment's own
        # paraphrases are the set its keep rule uses.
        reask_prompts = body.get("reask_prompts") or []
        if not isinstance(reask_prompts, list) or any(
            not isinstance(p, str) or not p.strip() for p in reask_prompts
        ):
            raise HTTPException(400, "reask_prompts must be a list of nonempty text.")
        # Entities that contain the expected answer's term but are still wrong.
        wrong_terms = body.get("wrong_terms") or []
        if not isinstance(wrong_terms, list) or any(
            not isinstance(w, str) or not w.strip() for w in wrong_terms
        ):
            raise HTTPException(400, "wrong_terms must be a list of nonempty text.")
        try:
            store.feedback(
                idx,
                thumbs == "down",
                note,
                reask_prompts=reask_prompts,
                wrong_terms=wrong_terms,
            )
        except KeyError as exc:
            raise HTTPException(404, "No such interaction.") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if thumbs == "down":
            controller.schedule()
        return {"interaction_idx": idx, "flagged": thumbs == "down"}

    @app.post("/trigger_review")
    async def trigger_review():
        pending = len(store.pending())
        if pending:
            controller.schedule(immediate=True)
        running = store.db.execute(
            "SELECT COUNT(*) FROM interactions WHERE status='reviewing'"
        ).fetchone()[0]
        return {
            "message": (
                "Review queued or running."
                if pending or running
                else "No pending feedback."
            ),
            "unreviewed_count": pending + running,
        }

    @app.get("/sync")
    async def sync():
        start = time.monotonic()
        results = await controller.sync()
        return dict(
            message="Reviews finished.",
            elapsed_time=time.monotonic() - start,
            tasks_count=len(results),
            reviews=results,
        )

    @app.get("/history")
    async def history():
        return {"history": store.rows()}

    @app.get("/status")
    async def status():
        return dict(
            status="failed" if controller.failed else "up",
            model=runtime.name,
            reviewing=controller.task is not None and not controller.task.done(),
            pending=len(store.pending()),
            error=controller.failed,
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [dict(id=runtime.name, object="model", owned_by="local")],
        }

    @app.get("/api/tags")
    async def tags():
        return {"models": [dict(name=runtime.name, model=runtime.name)]}

    @app.get("/api/version")
    async def version():
        if runtime.native:
            r = await runtime.client.get(runtime.url + "/api/version")
            r.raise_for_status()
            return r.json()
        return {"version": "adaptible"}

    @app.post("/api/show")
    async def show(request: Request):
        if not runtime.native:
            raise HTTPException(404, "Use /v1/models for this runtime.")
        body = await read_body(request)
        check_model(body)
        r = await runtime.client.post(
            runtime.url + "/api/show", json={"model": runtime.name}
        )
        r.raise_for_status()
        return r.json()

    return app
