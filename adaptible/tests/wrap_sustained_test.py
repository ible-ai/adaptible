"""Model-free checks for the bounded sustained recurrent protocol."""

import copy
import tempfile
import unittest
from unittest import mock

from scripts import wrapper_recurrent as recurrent


class SustainedOpportunityTest(unittest.TestCase):
    def run_fixture(self, *, required=4, full_budget=True, eligible=8, successes=4):
        cases = recurrent.CASES
        eligible_ids = {case[0] for case in cases[:eligible]}
        successful_ids = {case[0] for case in cases[:successes]}
        learned, history, queries, scopes = set(), [], {}, {}
        generation = dict(
            complete=True, thinking_observed=True, reasoning="Observed thought."
        )

        def routed(case_id):
            adapted = case_id in learned
            return {
                **generation,
                "adaptation": dict(
                    policy="correction_scoped_v1",
                    adapter=str(len(learned)) if adapted else "base",
                    adapter_sha256=f"{len(learned):064x}" if adapted else None,
                    scope=scopes[case_id] if adapted else None,
                ),
            }

        def matrix(*args, **kwargs):
            return [
                dict(
                    case_id=case[0],
                    prompt_index=index,
                    passed=case[0] in learned or case[0] not in eligible_ids,
                    classification=(
                        "correct"
                        if case[0] in learned or case[0] not in eligible_ids
                        else "factual_missing"
                    ),
                    response=(
                        case[1][0]
                        if case[0] in learned or case[0] not in eligible_ids
                        else "unknown"
                    ),
                    generation=routed(case[0]),
                )
                for case in cases
                for index in range(3)
            ]

        def chat(client, model, question, *, trace=None, **kwargs):
            case = next(
                case for case in cases if question == case[2] + recurrent.ANSWER_STYLE
            )
            idx = len(queries) + 1
            queries[idx] = case[0]
            if trace is not None:
                trace.update(generation)
            return "unknown", idx

        def request(client, method, path, **kwargs):
            if path == "/v1/models":
                data = {"data": [{"id": "fixture"}]}
            elif path == "/feedback":
                idx = kwargs["json"]["interaction_idx"]
                case_id = queries[idx]
                kept = case_id in successful_ids
                if kept:
                    learned.add(case_id)
                    scopes[case_id] = idx
                history.append(
                    dict(
                        interaction_idx=idx,
                        status="kept" if kept else "skipped",
                        note="",
                        references=dict(
                            kind="web",
                            sources=[dict(url="https://example.test/evidence")],
                        ),
                    )
                )
                data = {}
            elif path == "/sync":
                data = {}
            elif path == "/history":
                data = {"history": history}
            else:
                raise AssertionError(path)
            response = mock.Mock()
            response.json.return_value = copy.deepcopy(data)
            return response

        def artifact(state):
            index = len(learned)
            return dict(
                directory=f"/fixture/{index}",
                sha256=f"{index:064x}",
                previous=f"/fixture/{index - 1}/adapter" if index > 1 else None,
            )

        def stream(client, model, question, path, *, trace=None, **kwargs):
            case = next(
                case for case in cases if question == case[2] + recurrent.ANSWER_STYLE
            )
            if trace is not None:
                trace.update(routed(case[0]))
            return next(
                case[1][0]
                for case in cases
                if question == case[2] + recurrent.ANSWER_STYLE
            )

        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "llama-cpp",
                "/existing/model.gguf",
                "--thinking",
                "--cycles",
                "2",
                "--required-repairs",
                str(required),
                "--output-dir",
                directory,
            ]
            if full_budget:
                argv.append("--full-budget")
            args = recurrent.parser().parse_args(argv)
            with (
                mock.patch.object(recurrent, "WrapperProcess") as process,
                mock.patch.object(recurrent, "request", side_effect=request),
                mock.patch.object(recurrent, "evaluate", side_effect=matrix),
                mock.patch.object(recurrent, "chat", side_effect=chat),
                mock.patch.object(recurrent, "stream_chat", side_effect=stream),
                mock.patch.object(recurrent, "verify_training") as training,
                mock.patch.object(recurrent, "accepted_artifact", side_effect=artifact),
                mock.patch.object(
                    recurrent, "startup_base_digest", return_value="a" * 64
                ),
                mock.patch.object(
                    recurrent, "verify_final_base", return_value=dict(unchanged=True)
                ),
                mock.patch.object(recurrent, "cleanup_tags", return_value=[]),
                mock.patch.object(recurrent, "cleanup_lm_studio", return_value=[]),
            ):
                process.return_value.url = "http://fixture.test"
                report = recurrent.run_recurrent(args)
                return (
                    report,
                    process.return_value.start.call_count,
                    training.call_count,
                )

    def test_full_budget_visits_all_sixteen_slots_after_four_early_repairs(self):
        report, starts, trained = self.run_fixture()
        self.assertEqual(report["recurrent_status"], "passed", report)
        opportunities = report["review_opportunities"]
        self.assertEqual(
            [(row["cycle"], row["case_id"]) for row in opportunities],
            [(cycle, case[0]) for cycle in range(2) for case in recurrent.CASES],
        )
        self.assertEqual([row["status"] for row in opportunities[:4]], ["kept"] * 4)
        self.assertEqual(
            [row["status"] for row in opportunities[8:12]], ["already_repaired"] * 4
        )
        self.assertEqual(len(report["attempts"]), 12)
        self.assertEqual((starts, trained), (2, 4))
        self.assertEqual(len(report["streams"]), 4)

    def test_default_budget_stops_at_the_requested_four_repairs(self):
        report, _, trained = self.run_fixture(full_budget=False)
        self.assertEqual(report["recurrent_status"], "passed", report)
        self.assertEqual(len(report["review_opportunities"]), 4)
        self.assertEqual(trained, 4)

    def test_required_repairs_controls_initial_eligibility_gate(self):
        report, starts, trained = self.run_fixture(eligible=3, required=4)
        self.assertEqual(report["recurrent_status"], "inconclusive")
        self.assertIn("Fewer than 4", report["reason"])
        self.assertEqual(report["review_opportunities"], [])
        self.assertEqual((starts, trained), (1, 0))

    def test_required_repairs_controls_final_gate_without_weakening_proof(self):
        insufficient, starts, trained = self.run_fixture(required=4, successes=2)
        self.assertEqual(insufficient["recurrent_status"], "inconclusive")
        self.assertEqual(len(insufficient["accumulated_repairs"]), 2)
        self.assertEqual((starts, trained), (1, 2))
        sufficient, starts, trained = self.run_fixture(required=2, successes=2)
        self.assertEqual(sufficient["recurrent_status"], "passed", sufficient)
        self.assertEqual((starts, trained), (2, 2))


if __name__ == "__main__":
    unittest.main()
