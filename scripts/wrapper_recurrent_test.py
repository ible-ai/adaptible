"""Recurrent demo pass criteria; real native checks are explicitly opt-in."""

import copy
import json
import os
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import httpx

from scripts import wrapper_demo
from scripts import wrapper_recurrent as demo


class RecurrentAssessmentTest(unittest.TestCase):
    def test_source_fingerprint_uses_startup_identity_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            source = state / "original.gguf"
            source.write_bytes(b"original model bytes")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with closing(sqlite3.connect(state / "history.sqlite3")) as db:
                db.execute("CREATE TABLE meta (key TEXT, value TEXT)")
                db.execute(
                    "INSERT INTO meta VALUES (?, ?)",
                    ("identity", json.dumps("test:model:" + digest)),
                )
                db.commit()
            before = demo.startup_base_digest(state)
            self.assertEqual(before, digest)
            adapter = state / "adapters" / "candidate"
            adapter.mkdir(parents=True)
            (adapter / "job.json").write_text(json.dumps(dict(blob=str(source))))
            checked = demo.verify_final_base(state, before)
            self.assertTrue(checked["unchanged"])
            self.assertEqual(checked["before"], checked["after"])
            source.write_bytes(b"modified model bytes")
            with self.assertRaisesRegex(AssertionError, "base weights changed"):
                demo.verify_final_base(state, before)

    def test_lineage_can_include_two_candidates_in_one_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            middle = path / "middle"
            middle.mkdir()
            (middle / "job.json").write_text(
                json.dumps(dict(previous=str(path / "first" / "adapter")))
            )
            self.assertTrue(
                demo.continues_from(
                    dict(previous=str(middle / "adapter")),
                    dict(directory=str(path / "first")),
                )
            )
            self.assertFalse(
                demo.continues_from(
                    dict(previous=str(middle / "adapter")),
                    dict(directory=str(path / "unrelated")),
                )
            )

    def test_nonthinking_is_explicit_on_public_and_native_requests(self):
        bodies = []

        def respond(request):
            if request.url.path == "/history":
                return httpx.Response(
                    200,
                    json=dict(
                        history=[
                            dict(interaction_idx=1, status="new", response="Canberra")
                        ]
                    ),
                )
            body = json.loads(request.content)
            bodies.append(body)
            headers = {"X-Interaction-Idx": "1"}
            if not body.get("stream"):
                return httpx.Response(
                    200,
                    headers=headers,
                    json=dict(
                        model="test", choices=[dict(message=dict(content="Canberra"))]
                    ),
                )
            if request.url.path.startswith("/api/"):
                data = dict(model="test", done=True, response="Canberra")
                return httpx.Response(
                    200, headers=headers, text=json.dumps(data) + "\n"
                )
            data = dict(model="test", choices=[dict(delta=dict(content="Canberra"))])
            return httpx.Response(
                200,
                headers=headers,
                text="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n",
            )

        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            wrapper_demo.chat(client, "test", "Question?")
            self.assertNotIn("reasoning_effort", bodies[-1])
            wrapper_demo.chat(client, "test", "Question?", nonthinking=True)
            self.assertEqual(bodies[-1]["reasoning_effort"], "none")
            self.assertEqual(
                bodies[-1]["chat_template_kwargs"], dict(enable_thinking=False)
            )
            for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
                wrapper_demo.stream_chat(
                    client, "test", "Question?", path, nonthinking=True
                )
                if path.startswith("/api/"):
                    self.assertIs(bodies[-1]["think"], False)
                else:
                    self.assertEqual(bodies[-1]["reasoning_effort"], "none")
                    self.assertEqual(
                        bodies[-1]["chat_template_kwargs"], dict(enable_thinking=False)
                    )

    def test_whole_answer_and_factual_missing_are_distinct(self):
        for response, expected in (
            ("", "incomplete"),
            ("   ", "incomplete"),
            ("**...**", "incomplete"),
            (None, "incomplete"),
            ("**Casablanca**.", "correct"),
            ("</think>\nCasablanca", "correct"),
            ("Casablanca is the capital of Morocco.", "formatting_or_extra_claims"),
            ("Rabat, not Casablanca.", "formatting_or_extra_claims"),
            ("Rabat", "factual_missing"),
            ("<think>Casablanca", "incomplete"),
        ):
            with self.subTest(response=response):
                result = demo.assess(response, ("Casablanca",))
                self.assertEqual(result["classification"], expected)
                self.assertEqual(result["passed"], expected == "correct")

    def test_progress_uses_stderr_only_when_requested(self):
        args = demo.parser().parse_args(["ollama", "installed-model"])
        with mock.patch("sys.stderr") as stderr, mock.patch("sys.stdout") as stdout:
            demo.progress(args, "quiet by default")
            stderr.write.assert_not_called()
            args.progress = True
            demo.progress(args, "checking baseline")
            self.assertIn(
                "checking baseline",
                "".join(call.args[0] for call in stderr.write.call_args_list),
            )
            stdout.write.assert_not_called()

    def test_evaluation_answers_never_enter_chat_prompts(self):
        with mock.patch.object(demo, "chat", return_value=("unknown", 1)) as chat:
            matrix = demo.evaluate(None, "model")
        self.assertEqual(len(matrix), 24)
        for call in chat.call_args_list:
            prompt = call.args[2]
            self.assertFalse(
                any(alias in prompt for case in demo.CASES for alias in case[1])
            )


class RecurrentOrchestrationTest(unittest.TestCase):
    def run_fixture(self, *, regression=False, status="kept", web_sources=True):
        cases = demo.CASES[:2]
        baseline = [
            dict(
                case_id=case[0],
                prompt_index=index,
                passed=False,
                classification="factual_missing",
                response="wrong",
            )
            for case in cases
            for index in range(3)
        ]
        first = copy.deepcopy(baseline)
        for row in first[:3]:
            row.update(passed=True, classification="correct")
        second = copy.deepcopy(first)
        for row in second[3:]:
            row.update(passed=True, classification="correct")
        if regression:
            second[0].update(passed=False, classification="factual_missing")
        reviews = []

        def request(client, method, path, **kwargs):
            if path == "/v1/models":
                data = dict(data=[dict(id="fixture")])
            elif path == "/feedback":
                self.assertEqual(set(kwargs["json"]), {"interaction_idx", "thumbs"})
                idx = kwargs["json"]["interaction_idx"]
                reviews.append(
                    dict(
                        interaction_idx=idx,
                        status=status,
                        note="",
                        references=dict(
                            kind="web",
                            sources=(
                                [dict(url="https://example.org/fact")]
                                if web_sources
                                else []
                            ),
                        ),
                    )
                )
                data = {}
            elif path == "/sync":
                data = dict(reviews=reviews)
            elif path == "/history":
                data = dict(history=reviews)
            else:
                raise AssertionError(path)
            response = mock.Mock()
            response.json.return_value = copy.deepcopy(data)
            return response

        with tempfile.TemporaryDirectory() as directory:
            args = demo.parser().parse_args(
                ["vllm", "/existing/model", "--output-dir", directory, "--cycles", "1"]
            )
            artifacts = [
                dict(directory="/first", sha256="one", previous=None),
                dict(directory="/second", sha256="two", previous="/first/adapter"),
            ]
            with (
                mock.patch.object(demo, "CASES", cases),
                mock.patch.object(demo, "WrapperProcess") as process,
                mock.patch.object(demo, "request", side_effect=request),
                mock.patch.object(
                    demo, "evaluate", side_effect=[baseline, first, second, second]
                ),
                mock.patch.object(
                    demo, "chat", side_effect=[("wrong", 1), ("wrong", 2)]
                ),
                mock.patch.object(demo, "verify_training") as training,
                mock.patch.object(demo, "startup_base_digest", return_value="a" * 64),
                mock.patch.object(
                    demo, "verify_final_base", return_value=dict(unchanged=True)
                ),
                mock.patch.object(demo, "accepted_artifact", side_effect=artifacts),
                mock.patch.object(
                    demo, "stream_chat", side_effect=["Casablanca", "Dodoma"]
                ),
                mock.patch.object(demo, "cleanup_tags", return_value=[]),
                mock.patch.object(demo, "cleanup_lm_studio", return_value=[]),
            ):
                process.return_value.url = "http://test"
                report = demo.run_recurrent(args)
                self.assertTrue(args.web_search)
                self.assertFalse(
                    (Path(report["directory"]) / "documents.json").exists()
                )
                self.assertTrue((Path(report["directory"]) / "report.json").is_file())
                return (
                    report,
                    process.return_value.start.call_count,
                    training.call_count,
                )

    def test_two_distinct_trained_repairs_retention_restart_and_stream_are_required(
        self,
    ):
        report, starts, training = self.run_fixture()
        self.assertEqual(report["recurrent_status"], "passed", report)
        self.assertEqual(len(report["accumulated_repairs"]), 2)
        self.assertEqual((starts, training), (2, 2))
        self.assertEqual(len(report["streams"]), 2)

    def test_repair_that_forgets_previous_answer_cannot_pass(self):
        report, starts, _ = self.run_fixture(regression=True)
        self.assertEqual(report["recurrent_status"], "failed")
        self.assertIn("regressed", report["error"])
        self.assertEqual(starts, 1)

    def test_correct_answers_without_accepted_training_do_not_count(self):
        report, starts, training = self.run_fixture(status="rejected")
        self.assertEqual(report["recurrent_status"], "inconclusive")
        self.assertEqual(report["accumulated_repairs"], [])
        self.assertEqual((starts, training), (1, 0))

    def test_kept_without_automatic_evidence_cannot_pass(self):
        report, _, training = self.run_fixture(web_sources=False)
        self.assertEqual(report["recurrent_status"], "failed")
        self.assertIn("no web evidence", report["error"])
        self.assertEqual(training, 0)


class RecurrentLiveTest(unittest.TestCase):
    def run_backend(self, service, variable, flag=None, executable=None):
        model = os.environ.get(variable)
        if not model:
            self.skipTest(f"Set {variable} to run native recurrent learning")
        thinking = os.environ.get("ADAPTIBLE_RECURRENT_THINKING") == "1"
        argv = [service, model, "--thinking" if thinking else "--nonthinking"]
        if thinking:
            argv += ["--timeout", "1200"]
        if os.environ.get("ADAPTIBLE_RECURRENT_OUTPUT_DIR"):
            argv += ["--output-dir", os.environ["ADAPTIBLE_RECURRENT_OUTPUT_DIR"]]
        if flag and os.environ.get(executable):
            argv += [flag, os.environ[executable]]
        report = demo.run_recurrent(demo.parser().parse_args(argv))
        self.assertEqual(
            report["recurrent_status"], "passed", json.dumps(report, indent=2)
        )

    def test_ollama(self):
        self.run_backend("ollama", "ADAPTIBLE_RECURRENT_OLLAMA_MODEL")

    def test_llama_cpp(self):
        self.run_backend(
            "llama-cpp",
            "ADAPTIBLE_RECURRENT_LLAMA_GGUF",
            "--llama-server",
            "ADAPTIBLE_TEST_LLAMA_SERVER",
        )

    def test_lm_studio(self):
        self.run_backend(
            "lm-studio",
            "ADAPTIBLE_RECURRENT_LM_STUDIO_GGUF",
            "--lms",
            "ADAPTIBLE_TEST_LMS",
        )

    def test_vllm(self):
        self.run_backend(
            "vllm",
            "ADAPTIBLE_RECURRENT_VLLM_MODEL",
            "--vllm-server",
            "ADAPTIBLE_VLLM_EXECUTABLE",
        )


if __name__ == "__main__":
    unittest.main()
