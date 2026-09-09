"""HTML report generator for evaluation results."""

import html
import json
from pathlib import Path

from .harness import EvaluationResult


def generate_html_report(
    result: EvaluationResult,
    output_path: str | Path = "/tmp/adaptible_eval_report.html",
) -> str:
    """Generate an HTML report from evaluation results."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build item cards
    train_items_html = []
    holdout_items_html = []

    for item in result.items:
        initial_class = "has-answer" if item.initial_has_key_terms else "missing-answer"
        post_class = "has-answer" if item.post_has_key_terms else "missing-answer"

        # Determine change status
        if item.initial_has_key_terms and item.post_has_key_terms:
            status = "retained"
            status_text = "Retained"
            status_icon = "="
        elif not item.initial_has_key_terms and item.post_has_key_terms:
            status = "improved"
            status_text = "Improved"
            status_icon = "+"
        elif item.initial_has_key_terms and not item.post_has_key_terms:
            status = "regressed"
            status_text = "Regressed"
            status_icon = "-"
        else:
            status = "unchanged"
            status_text = "No Change"
            status_icon = "·"

        if item.revision_invalid:
            revision_html = (
                '<div class="key-terms"><strong>Revision:</strong> '
                "INVALID (item skipped, not trained)</div>"
            )
        elif item.revision_text is not None:
            revision_html = (
                '<div class="key-terms"><strong>Trained on (self-generated):</strong> '
                f"{html.escape(item.revision_text[:300])}</div>"
            )
        else:
            revision_html = ""

        # Self-generated runs: show the parsed revision (what was trained on)
        # between the baseline and post-training answers, judged on its own.
        if item.revision_answer is not None:
            rev_ok = bool(item.revision_has_key_terms)
            rev_class = "has-answer" if rev_ok else "missing-answer"
            if item.revision_changed_text is False:
                rev_note = "restates baseline"
            elif item.revision_fixed:
                rev_note = "fixes baseline"
            elif item.revision_broke:
                rev_note = "breaks baseline"
            else:
                rev_note = "same verdict"
            revision_box_html = f"""
                <div class="response-box revision {rev_class}">
                    <div class="response-label">
                        Revision {'✓' if rev_ok else '✗'} <small>({rev_note})</small>
                    </div>
                    <div class="response-content">{html.escape(item.revision_answer[:500])}{'...' if len(item.revision_answer) > 500 else ''}</div>
                </div>"""
            responses_class = "responses-container three"
        else:
            revision_box_html = ""
            responses_class = "responses-container"

        card_html = f"""
        <div class="item-card">
            <div class="item-header">
                <span class="item-id">{html.escape(item.item_id)}</span>
                <span class="status-badge {status}">{status_icon} {status_text}</span>
            </div>

            <div class="question-box">
                <strong>Q:</strong> {html.escape(item.question)}
            </div>

            <div class="answer-box">
                <strong>Expected:</strong> {html.escape(item.correct_answer)}
                <div class="key-terms">Key terms: {html.escape(', '.join(item.key_terms))}</div>
                {revision_html}
            </div>

            <div class="{responses_class}">
                <div class="response-box {initial_class}">
                    <div class="response-label">
                        Initial {'✓' if item.initial_has_key_terms else '✗'}
                    </div>
                    <div class="response-content">{html.escape(item.initial_response[:500])}{'...' if len(item.initial_response) > 500 else ''}</div>
                </div>{revision_box_html}
                <div class="response-box {post_class}">
                    <div class="response-label">
                        Post-Training {'✓' if item.post_has_key_terms else '✗'}
                    </div>
                    <div class="response-content">{html.escape((item.post_response or '')[:500])}{'...' if item.post_response and len(item.post_response) > 500 else ''}</div>
                </div>
            </div>
        </div>
        """

        if item.was_trained:
            train_items_html.append(card_html)
        else:
            holdout_items_html.append(card_html)

    # Count statistics for summary
    train_items = result.train_items
    holdout_items = result.holdout_items

    train_improved = sum(
        1 for i in train_items if not i.initial_has_key_terms and i.post_has_key_terms
    )
    train_retained = sum(
        1 for i in train_items if i.initial_has_key_terms and i.post_has_key_terms
    )
    train_regressed = sum(
        1 for i in train_items if i.initial_has_key_terms and not i.post_has_key_terms
    )
    train_unchanged = sum(
        1
        for i in train_items
        if not i.initial_has_key_terms and not i.post_has_key_terms
    )

    holdout_improved = sum(
        1 for i in holdout_items if not i.initial_has_key_terms and i.post_has_key_terms
    )
    holdout_retained = sum(
        1 for i in holdout_items if i.initial_has_key_terms and i.post_has_key_terms
    )
    holdout_regressed = sum(
        1 for i in holdout_items if i.initial_has_key_terms and not i.post_has_key_terms
    )

    training_source = getattr(result.config, "training_source", "ground_truth")
    revision_prompt = getattr(result.config, "revision_prompt", "default")
    think_mode = getattr(result.config, "think_mode", "empty")
    rehearsal_k = getattr(result.config, "rehearsal_k", 0)
    rehearsal_max_tokens = getattr(result.config, "rehearsal_max_tokens", 768)
    collapse_text = html.escape(result.collapse_summary_text())
    if training_source == "self_generated":
        training_source_text = (
            "the model was trained on its <em>own revision</em> of each baseline "
            "answer (self-correction). Nothing here was trained on the label."
        )
    else:
        training_source_text = (
            "the model was fine-tuned directly on the <em>ground-truth label</em>. "
            "These numbers measure absorbing a supplied correction, "
            "<strong>not</strong> self-correction."
        )
    revision_invalid_count = result.revision_invalid_count
    revision_invalid_text = (
        f" {revision_invalid_count} item(s) were skipped because the revision "
        "failed validation."
        if revision_invalid_count
        else ""
    )
    revision_summary_html = (
        f'<div class="revision-summary">{html.escape(result.revision_summary_text())}'
        " Judged before training, so this is the quality of what the model had to "
        "learn from.</div>"
        if training_source == "self_generated"
        else ""
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptible Evaluation Report - {html.escape(result.config.name)} [{html.escape(training_source)}]</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
        h3 {{ color: #666; margin-top: 30px; }}

        .config-box {{
            background: #e9ecef;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 0.9em;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .metric-card.good .metric-value {{ color: #28a745; }}
        .metric-card.warning .metric-value {{ color: #ffc107; }}
        .metric-card.bad .metric-value {{ color: #dc3545; }}

        .breakdown-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }}
        .breakdown-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .breakdown-item.improved {{ border-left: 4px solid #28a745; }}
        .breakdown-item.retained {{ border-left: 4px solid #007bff; }}
        .breakdown-item.regressed {{ border-left: 4px solid #dc3545; }}
        .breakdown-item.unchanged {{ border-left: 4px solid #6c757d; }}
        .breakdown-count {{ font-size: 1.5em; font-weight: bold; }}

        .item-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .item-header {{
            background: #007bff;
            color: white;
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .item-id {{ font-weight: bold; }}
        .status-badge {{
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .status-badge.improved {{ background: #28a745; }}
        .status-badge.retained {{ background: #17a2b8; }}
        .status-badge.regressed {{ background: #dc3545; }}
        .status-badge.unchanged {{ background: #6c757d; }}

        .question-box, .answer-box {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .answer-box {{ background: #f8f9fa; }}
        .key-terms {{ font-size: 0.85em; color: #666; margin-top: 5px; }}

        .responses-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
        }}
        .responses-container.three {{ grid-template-columns: 1fr 1fr 1fr; }}
        .response-box.revision {{ background: #fffbea; }}
        .response-label small {{ font-weight: normal; color: #666; }}
        .revision-summary {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(0,0,0,0.1);
            font-size: 0.95em;
        }}
        .response-box {{
            padding: 15px;
            border-right: 1px solid #eee;
        }}
        .response-box:last-child {{ border-right: none; }}
        .response-label {{
            font-weight: bold;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }}
        .response-content {{
            font-family: monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }}
        .has-answer .response-label {{ color: #28a745; }}
        .missing-answer .response-label {{ color: #dc3545; }}

        .timestamp {{ color: #666; font-size: 0.9em; }}

        .training-source {{
            font-size: 1.15em;
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 6px solid;
        }}
        .training-source.ground_truth {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #664d03;
        }}
        .training-source.self_generated {{
            background: #d1e7dd;
            border-color: #198754;
            color: #0f5132;
        }}
        .training-source code {{ font-size: 1.1em; font-weight: bold; }}

        details {{ margin-top: 20px; }}
        details summary {{ cursor: pointer; color: #007bff; font-weight: bold; }}
        details pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85em;
        }}

        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .section-count {{
            background: #007bff;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Adaptible Evaluation Report <small>({html.escape(training_source)})</small></h1>
    <p class="timestamp">Generated: {result.timestamp} | Duration: {result.total_time_seconds:.1f}s</p>

    <div class="training-source {html.escape(training_source)}">
        <strong>Training source:</strong> <code>{html.escape(training_source)}</code>
        (revision prompt: <code>{html.escape(revision_prompt)}</code>) &mdash;
        {training_source_text}{revision_invalid_text}
        {revision_summary_html}
    </div>

    <div class="config-box">
        <strong>Configuration:</strong> {html.escape(result.config.name)}<br>
        Dataset: {html.escape(result.dataset_name)} ({len(result.items)} items)<br>
        Training source: {html.escape(training_source)} |
        Revision prompt: {html.escape(revision_prompt)} |
        Think mode: <code>{html.escape(str(think_mode))}</code> |
        Rehearsal k: <code>{rehearsal_k}</code> (max tokens: <code>{rehearsal_max_tokens}</code>) |
        Training iterations: {result.config.training_iterations} |
        Train/Holdout split: {result.config.train_ratio:.0%}/{1-result.config.train_ratio:.0%} |
        Shuffle: {result.config.shuffle} (seed: {result.config.seed})
    </div>

    <h2>Overall Metrics</h2>
    <div class="config-box">{collapse_text}<br>{html.escape(result.holdout_summary_text())}</div>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{result.baseline_accuracy:.0%}</div>
            <div class="metric-label">Baseline Accuracy<br>(all items)</div>
        </div>
        <div class="metric-card {'good' if result.train_post_accuracy > result.baseline_accuracy else ''}">
            <div class="metric-value">{result.train_post_accuracy:.0%}</div>
            <div class="metric-label">Train Post-Accuracy</div>
        </div>
        <div class="metric-card {'good' if result.train_improvement_rate > 0.5 else 'warning' if result.train_improvement_rate > 0.2 else ''}">
            <div class="metric-value">{result.train_improvement_rate:.0%}</div>
            <div class="metric-label">Train Improvement Rate<br>(wrong → right)</div>
        </div>
        <div class="metric-card {'good' if result.train_retention_rate > 0.8 else 'warning' if result.train_retention_rate > 0.5 else 'bad'}">
            <div class="metric-value">{result.train_retention_rate:.0%}</div>
            <div class="metric-label">Train Retention Rate<br>(stayed right)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{result.holdout_accuracy:.0%}</div>
            <div class="metric-label">Holdout Accuracy<br>(generalization)</div>
        </div>
    </div>

    <h2>Training Set Breakdown</h2>
    <div class="breakdown-grid">
        <div class="breakdown-item improved">
            <div class="breakdown-count">{train_improved}</div>
            <div>Improved</div>
        </div>
        <div class="breakdown-item retained">
            <div class="breakdown-count">{train_retained}</div>
            <div>Retained</div>
        </div>
        <div class="breakdown-item regressed">
            <div class="breakdown-count">{train_regressed}</div>
            <div>Regressed</div>
        </div>
        <div class="breakdown-item unchanged">
            <div class="breakdown-count">{train_unchanged}</div>
            <div>Unchanged</div>
        </div>
    </div>

    <h2>Holdout Set Breakdown</h2>
    <div class="breakdown-grid">
        <div class="breakdown-item improved">
            <div class="breakdown-count">{holdout_improved}</div>
            <div>Improved</div>
        </div>
        <div class="breakdown-item retained">
            <div class="breakdown-count">{holdout_retained}</div>
            <div>Retained</div>
        </div>
        <div class="breakdown-item regressed">
            <div class="breakdown-count">{holdout_regressed}</div>
            <div>Regressed</div>
        </div>
        <div class="breakdown-item unchanged">
            <div class="breakdown-count">{len(holdout_items) - holdout_improved - holdout_retained - holdout_regressed}</div>
            <div>Unchanged</div>
        </div>
    </div>

    <h2 class="section-header">
        <span>Training Set Items</span>
        <span class="section-count">{len(train_items)} items</span>
    </h2>
    {''.join(train_items_html)}

    <h2 class="section-header">
        <span>Holdout Set Items</span>
        <span class="section-count">{len(holdout_items)} items</span>
    </h2>
    {''.join(holdout_items_html)}

    <h2>Raw Data</h2>
    <details>
        <summary>JSON Data</summary>
        <pre>{html.escape(json.dumps(result.to_dict(), indent=2, default=str))}</pre>
    </details>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_content)

    return str(output_path)
