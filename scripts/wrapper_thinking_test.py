"""Independent model-free proof that thinking cannot substitute for an answer."""

import json
import unittest

import httpx

from scripts import wrapper_demo as demo
from scripts import wrapper_recurrent as recurrent


class HarnessHTTPErrorTest(unittest.TestCase):
    def test_native_error_body_retains_http_exception_and_response(self):
        response = httpx.Response(400, json={"error": "this model requires boolean thinking"})
        with httpx.Client(
            base_url="http://wrapper.invalid",
            transport=httpx.MockTransport(lambda request: response),
        ) as client:
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                demo.request(client, "POST", "/v1/chat/completions",
                             headers={"Authorization": "Bearer private-header"},
                             json={"messages": "private-request-body"})
        error = caught.exception
        self.assertIn("this model requires boolean thinking", str(error))
        self.assertIs(error.response, response)
        self.assertIs(error.request, response.request)
        self.assertNotIn("private-header", str(error))
        self.assertNotIn("private-request-body", str(error))

    def test_native_error_excerpt_is_bounded_to_two_thousand_characters(self):
        response = httpx.Response(422, text="x" * 2000 + "private-trailing-text")
        with httpx.Client(
            base_url="http://wrapper.invalid",
            transport=httpx.MockTransport(lambda request: response),
        ) as client:
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                demo.request(client, "POST", "/v1/chat/completions")
        message = str(caught.exception)
        self.assertIn("Response body: " + "x" * 2000 + " [truncated]", message)
        self.assertNotIn("private-trailing-text", message)


class ThinkingFramingTest(unittest.TestCase):
    def test_closed_native_and_prefilled_thoughts_have_separate_final(self):
        for text in (
            "<think>Consider evidence.</think>\nCasablanca.",
            "Consider evidence.</think>\nCasablanca.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    demo.split_thinking(text),
                    ("Casablanca.", "Consider evidence.", True),
                )

    def test_plain_answer_is_not_evidence_of_thinking(self):
        trace = demo.response_trace("Casablanca.", finish_reason="stop")
        self.assertTrue(trace["complete"])
        self.assertFalse(trace["thinking_observed"])
        self.assertEqual(trace["reasoning_chars"], 0)

    def test_empty_think_block_is_not_evidence_of_thinking(self):
        trace = demo.response_trace(
            "<think>\n\n</think>\nCasablanca.", finish_reason="stop"
        )
        self.assertTrue(trace["complete"])
        self.assertFalse(trace["thinking_observed"])

    def test_malformed_framing_cannot_launder_a_correct_suffix(self):
        for text in (
            "<think>unfinished Casablanca",
            "Rabat<think>reconsider</think>Casablanca",
            "<think>one<think>two</think>Casablanca",
            "<think>one</think>Rabat</think>Casablanca",
            "<think>one</think><think>two</think>Casablanca",
        ):
            with self.subTest(text=text):
                self.assertFalse(demo.split_thinking(text)[2])
                trace = demo.response_trace(text, finish_reason="stop")
                self.assertFalse(trace["framing_valid"])
                self.assertFalse(trace["complete"])
                self.assertFalse(recurrent.assess(text, ("Casablanca",))["passed"])

    def test_alias_inside_thought_does_not_repair_wrong_final(self):
        result = recurrent.assess(
            "<think>Casablanca is a candidate.</think>Rabat.", ("Casablanca",)
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["classification"], "factual_missing")

    def test_reasoning_without_final_is_incomplete(self):
        for content, reasoning in (
            ("", "Casablanca?"),
            ("<think>Casablanca?</think>", ""),
        ):
            with self.subTest(content=content):
                trace = demo.response_trace(content, reasoning, finish_reason="stop")
                self.assertTrue(trace["thinking_observed"])
                self.assertFalse(trace["complete"])

    def test_correct_text_with_truncation_is_not_complete(self):
        for reason in ("length", "max_tokens", "tool_calls", None):
            with self.subTest(reason=reason):
                self.assertFalse(
                    demo.response_trace(
                        "Casablanca.", "Reason here.", finish_reason=reason
                    )["complete"]
                )

    def test_token_count_without_captured_thought_is_diagnostic_only(self):
        usage = {"completion_tokens_details": {"reasoning_tokens": 12}}
        trace = demo.response_trace("Casablanca.", finish_reason="stop", usage=usage)
        self.assertEqual(trace["reasoning_tokens"], 12)
        self.assertEqual(trace["usage"], usage)
        self.assertFalse(trace["thinking_observed"])

    def test_recurrent_oracle_marks_truncated_correct_text_incomplete(self):
        trace = demo.response_trace(
            "Casablanca.", "Reason here.", finish_reason="length"
        )
        result = recurrent.assess_generation(
            "Casablanca.", ("Casablanca",), trace, thinking=True
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["classification"], "incomplete")

    def test_complete_repair_gate_requires_thinking_for_every_wording(self):
        good = {"generation": {"complete": True, "thinking_observed": True}}
        recurrent.require_thinking([good, good, good], "missing reasoning")
        for missing in (
            {},
            {"generation": {"complete": True, "thinking_observed": False}},
            {"generation": {"complete": False, "thinking_observed": True}},
        ):
            for position in range(3):
                with self.subTest(missing=missing, position=position):
                    rows = [good, good, good]
                    rows[position] = missing
                    with self.assertRaises(AssertionError):
                        recurrent.require_thinking(rows, "missing reasoning")


class ThinkingTransportTest(unittest.TestCase):
    def test_chat_explicitly_enables_thinking_and_preserves_separate_fields(self):
        for field in ("reasoning_content", "reasoning"):
            with self.subTest(field=field):
                bodies = []

                def respond(request):
                    bodies.append(json.loads(request.content))
                    return httpx.Response(
                        200,
                        headers={
                            "X-Interaction-Idx": "7",
                            "X-Adaptible-Adapter": "base",
                        },
                        json={
                            "model": "test",
                            "choices": [
                                {
                                    "message": {
                                        "content": "Rabat.",
                                        field: "Casablanca?",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"completion_tokens": 23},
                        },
                    )

                with httpx.Client(
                    base_url="http://test", transport=httpx.MockTransport(respond)
                ) as client:
                    trace = {}
                    answer, idx = demo.chat(
                        client,
                        "test",
                        "Question?",
                        thinking=True,
                        max_tokens=2048,
                        trace=trace,
                    )
                self.assertEqual((answer, idx), ("Rabat.", 7))
                self.assertEqual(bodies[0]["reasoning_effort"], "high")
                self.assertIs(
                    bodies[0]["chat_template_kwargs"]["enable_thinking"], True
                )
                self.assertEqual(bodies[0]["max_tokens"], 2048)
                self.assertEqual(trace["reasoning"], "Casablanca?")
                self.assertEqual(trace["final"], "Rabat.")
                self.assertEqual(
                    trace["adaptation_headers"], dict(adapter="base", scope=None)
                )
                self.assertTrue(trace["complete"])
                self.assertFalse(recurrent.assess(answer, ("Casablanca",))["passed"])

    def stream_result(
        self,
        path,
        *,
        finish="stop",
        inline=False,
        done=True,
        route=None,
        native_ollama=False,
    ):
        native = path.startswith("/api/")
        chunks = (
            ["<thi", "nk>Reason", " here.</thi", "nk>", "Casablanca."]
            if inline
            else ["Casablanca."]
        )
        stored, bodies = demo.split_thinking("".join(chunks))[0], []

        def respond(request):
            if request.url.path == "/history":
                return httpx.Response(
                    200,
                    json={
                        "history": [
                            {
                                "interaction_idx": 7,
                                "status": "new",
                                "response": stored,
                                "response_details": {"adaptation": route},
                            }
                        ]
                    },
                )
            bodies.append(json.loads(request.content))
            events = []
            if native:
                if path == "/api/chat":
                    events.append(
                        {"model": "test", "message": {"thinking": "Reason here."}}
                    )
                    events.extend(
                        {"model": "test", "message": {"content": s}} for s in chunks
                    )
                else:
                    events.append({"model": "test", "thinking": "Reason here."})
                    events.extend({"model": "test", "response": s} for s in chunks)
                events.append(
                    {
                        "model": "test",
                        "done": done,
                        "done_reason": finish,
                        "eval_count": 19,
                    }
                )
                text = "\n".join(json.dumps(event) for event in events) + "\n"
            else:
                if not inline:
                    events.append(
                        {
                            "model": "test",
                            "choices": [
                                {"delta": {"reasoning_content": "Reason here."}}
                            ],
                        }
                    )
                events.extend(
                    {"model": "test", "choices": [{"delta": {"content": s}}]}
                    for s in chunks
                )
                events.append(
                    {
                        "model": "test",
                        "choices": [{"delta": {}, "finish_reason": finish}],
                    }
                )
                events.append(
                    {
                        "model": "test",
                        "choices": [],
                        "usage": {"completion_tokens_details": {"reasoning_tokens": 3}},
                    }
                )
                text = "".join(
                    "data: " + json.dumps(event) + "\n\n" for event in events
                )
                if done:
                    text += "data: [DONE]\n\n"
            headers = {"X-Interaction-Idx": "7"}
            if route is not None:
                headers["X-Adaptible-Adapter"] = route["adapter"]
                if route.get("scope") is not None:
                    headers["X-Adaptible-Scope"] = str(route["scope"])
            return httpx.Response(200, headers=headers, text=text)

        trace = {}
        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            result = demo.stream_chat(
                client,
                "test",
                "Question?",
                path,
                thinking=True,
                max_tokens=2048,
                trace=trace,
                native_ollama=native_ollama,
            )
        return result, trace, bodies[0]

    def test_public_and_native_streams_separate_thought_from_final(self):
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            with self.subTest(path=path):
                result, trace, body = self.stream_result(path)
                self.assertEqual(result, "Casablanca.")
                self.assertEqual(trace["reasoning"], "Reason here.")
                self.assertEqual(trace["final"], "Casablanca.")
                self.assertTrue(trace["thinking_observed"])
                self.assertTrue(trace["complete"])
                if path.startswith("/api/"):
                    self.assertIs(body["think"], True)
                    self.assertEqual(body["options"]["num_predict"], 2048)
                else:
                    self.assertEqual(body["reasoning_effort"], "high")
                    self.assertIs(body["chat_template_kwargs"]["enable_thinking"], True)
                    self.assertEqual(body["max_tokens"], 2048)

    def test_split_inline_delimiters_are_parsed_after_stream_reassembly(self):
        result, trace, _ = self.stream_result("/v1/chat/completions", inline=True)
        self.assertIn("<think>", result)
        self.assertEqual(trace["reasoning"], "Reason here.")
        self.assertEqual(trace["final"], "Casablanca.")
        self.assertTrue(trace["framing_valid"])
        self.assertTrue(trace["complete"])

    def test_done_event_cannot_hide_length_truncation(self):
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            with self.subTest(path=path):
                self.assertFalse(
                    self.stream_result(path, finish="length")[1]["complete"]
                )

    def test_stream_without_terminal_event_is_rejected(self):
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            with self.subTest(path=path), self.assertRaises(AssertionError):
                self.stream_result(path, done=False)

    def test_conflicting_mode_flags_fail_before_transport(self):
        with httpx.Client(base_url="http://test") as client:
            with self.assertRaises(AssertionError):
                demo.chat(client, "test", "Question?", thinking=True, nonthinking=True)
            with self.assertRaises(AssertionError):
                demo.stream_chat(
                    client,
                    "test",
                    "Question?",
                    "/api/chat",
                    thinking=True,
                    nonthinking=True,
                )

    def test_all_stream_protocols_preserve_recorded_adapter_identity(self):
        route = dict(
            policy="correction_scoped_v1",
            adapter="candidate",
            adapter_sha256="a" * 64,
            scope=7,
        )
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            with self.subTest(path=path):
                _, trace, _ = self.stream_result(path, route=route)
                self.assertEqual(trace["adaptation"], route)
                recurrent.require_route(
                    [{"generation": trace}],
                    dict(directory="/state/candidate", sha256="a" * 64),
                    scopes={7},
                )

    def test_ollama_openai_mode_omits_unsupported_template_controls(self):
        bodies = []

        def respond(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"X-Interaction-Idx": "1"},
                json=dict(
                    model="test",
                    choices=[
                        dict(message=dict(content="Answer"), finish_reason="stop")
                    ],
                ),
            )

        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            for thinking in (False, True):
                demo.chat(
                    client,
                    "test",
                    "Question?",
                    thinking=thinking,
                    nonthinking=not thinking,
                    native_ollama=True,
                )
                self.assertNotIn("chat_template_kwargs", bodies[-1])
                self.assertEqual(
                    bodies[-1]["reasoning_effort"], "high" if thinking else "none"
                )
        _, _, body = self.stream_result("/v1/chat/completions", native_ollama=True)
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(body["reasoning_effort"], "high")


class ScopedDeploymentProofTest(unittest.TestCase):
    def row(self, *, adapter="new", digest="a" * 64, scope=7, case="learned"):
        return dict(
            case_id=case,
            passed=True,
            generation=dict(
                complete=True,
                thinking_observed=True,
                adaptation=dict(
                    policy="correction_scoped_v1",
                    adapter=adapter,
                    adapter_sha256=digest,
                    scope=scope,
                ),
            ),
        )

    def test_correct_answer_on_base_or_old_adapter_cannot_count_as_learning(self):
        artifact = dict(directory="/state/new", sha256="a" * 64)
        recurrent.require_route([self.row()], artifact, scopes={7})
        for row in (
            self.row(adapter="base", digest=None, scope=None),
            self.row(adapter="old"),
            self.row(digest="b" * 64),
            self.row(scope=8),
            self.row(scope=True),
            {"passed": True},
        ):
            with self.subTest(row=row), self.assertRaises(AssertionError):
                recurrent.require_route([row], artifact, scopes={7})

    def test_baseline_and_unrelated_routes_must_have_no_adapter_or_scope(self):
        recurrent.require_route([self.row(adapter="base", digest=None, scope=None)])
        for row in (
            self.row(),
            self.row(adapter="base", scope=None),
            self.row(adapter="base", digest=None, scope=7),
        ):
            with self.subTest(row=row), self.assertRaises(AssertionError):
                recurrent.require_route([row])

    def test_previous_learned_scope_must_use_newest_cumulative_adapter(self):
        artifact = dict(directory="/state/new", sha256="a" * 64)
        scopes = {"first": {7}, "second": {9}}
        rows = [
            self.row(case="first"),
            self.row(case="second", scope=9),
            self.row(case="unrelated", adapter="base", digest=None, scope=None),
        ]
        recurrent.require_deployment(rows, artifact, scopes, {"first", "second"})
        for index, replacement in (
            (0, self.row(case="first", adapter="old")),
            (2, self.row(case="unrelated")),
        ):
            broken = list(rows)
            broken[index] = replacement
            with self.subTest(index=index), self.assertRaises(AssertionError):
                recurrent.require_deployment(
                    broken, artifact, scopes, {"first", "second"}
                )

    def test_headers_must_match_durable_route(self):
        route = self.row()["generation"]["adaptation"]
        row = {"response_details": {"adaptation": route}}
        trace = {"adaptation_headers": dict(adapter="new", scope=7)}
        demo.record_adaptation(trace, row)
        self.assertEqual(trace["adaptation"], route)
        for headers in (
            None,
            dict(adapter="old", scope=7),
            dict(adapter="new", scope=8),
        ):
            with self.subTest(headers=headers), self.assertRaises(AssertionError):
                demo.record_adaptation({"adaptation_headers": headers}, row)

    def test_scoped_matrix_collects_durable_routes_in_one_batch(self):
        from unittest import mock

        route = dict(
            policy="correction_scoped_v1",
            adapter="base",
            adapter_sha256=None,
            scope=None,
        )
        history = []

        def chat(client, model, question, *, trace, **kwargs):
            idx = len(history) + 1
            history.append(
                dict(interaction_idx=idx, response_details={"adaptation": route})
            )
            trace["adaptation_headers"] = dict(adapter="base", scope=None)
            return "unknown", idx

        response = mock.Mock()
        response.json.side_effect = lambda: {"history": history}
        with (
            mock.patch.object(recurrent, "chat", side_effect=chat),
            mock.patch.object(recurrent, "request", return_value=response) as request,
        ):
            rows = recurrent.evaluate(None, "model", scoped=True)
        self.assertEqual(len(rows), 24)
        request.assert_called_once_with(None, "GET", "/history")
        recurrent.require_route(rows)


if __name__ == "__main__":
    unittest.main()
