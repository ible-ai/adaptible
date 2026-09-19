"""Compare previous repairs under the immediately current and candidate weights.

This is a wrapper acceptance guard, separate from the deliberate local-greedy
policy in the historical cycle experiments. Preserve each passing prompt rather
than a summed score: a newly passing paraphrase cannot conceal another's loss.
"""

from .thinking import thinking_complete


async def evaluate_retention(repairs, runtime, judge, *, handle=None):
    """Re-ask recorded prompts without inserting references into model input."""
    results = []

    for repair in repairs:
        thinking = (repair.get("generation_mode") or {}).get("thinking", False)
        # Held-out prompts must have been selected before evaluating candidates;
        # they are stored separately so training can avoid consuming them.
        prompts = repair.get("prompts") or [repair["messages"]]
        for kind, messages_list in (
            ("repair", prompts),
            ("heldout", repair.get("heldout_prompts", [])),
        ):
            for index, messages in enumerate(messages_list):
                generation = {}
                mode_options = (
                    dict(thinking=True, details=generation) if thinking else {}
                )
                response = await runtime.complete(
                    messages, handle=handle, **mode_options
                )
                passed = (
                    not thinking or thinking_complete(generation)
                ) and await judge(
                    repair["question"],
                    response,
                    repair["expected"],
                    note=repair.get("note"),
                )
                results.append(
                    dict(
                        interaction_idx=repair["interaction_idx"],
                        kind=kind,
                        prompt_index=index,
                        messages=messages,
                        response=response,
                        expected=repair["expected"],
                        passed=bool(passed),
                        generation=generation,
                        thinking=thinking,
                    )
                )
    return results


def retention_regressions(before, after):
    """Return each formerly passing check that failed or disappeared."""

    def key(result):
        return result["interaction_idx"], result["kind"], result["prompt_index"]

    after_by_key = {key(result): result for result in after}
    return [
        result
        for result in before
        if result["passed"]
        and (
            key(result) not in after_by_key
            or not after_by_key[key(result)]["passed"]
            or after_by_key[key(result)]["messages"] != result["messages"]
        )
    ]
