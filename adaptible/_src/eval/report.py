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
                <div class="key-terms">Key terms: {', '.join(item.key_terms)}</div>
            </div>

            <div class="responses-container">
                <div class="response-box {initial_class}">
                    <div class="response-label">
                        Initial {'✓' if item.initial_has_key_terms else '✗'}
                    </div>
                    <div class="response-content">{html.escape(item.initial_response[:500])}{'...' if len(item.initial_response) > 500 else ''}</div>
                </div>
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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptible Evaluation Report - {html.escape(result.config.name)}</title>
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
    <h1>Adaptible Evaluation Report</h1>
    <p class="timestamp">Generated: {result.timestamp} | Duration: {result.total_time_seconds:.1f}s</p>

    <div class="config-box">
        <strong>Configuration:</strong> {html.escape(result.config.name)}<br>
        Dataset: {html.escape(result.dataset_name)} ({len(result.items)} items)<br>
        Training iterations: {result.config.training_iterations} |
        Train/Holdout split: {result.config.train_ratio:.0%}/{1-result.config.train_ratio:.0%} |
        Shuffle: {result.config.shuffle} (seed: {result.config.seed})
    </div>

    <h2>Overall Metrics</h2>
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
