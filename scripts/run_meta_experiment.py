#!/usr/bin/env python3
"""Run a meta-learning experiment.

This script runs the meta-learning infrastructure to compare learning trajectories
across multiple model instances with different random seeds.

Usage:
    python scripts/run_meta_experiment.py [options]

Options:
    --name NAME             Experiment name (default: "meta_experiment")
    --seeds SEEDS           Comma-separated seeds (default: "42,123,456")
    --checkpoint-interval N Checkpoint every N training events (default: 10)
    --iterations N          Step cap per training call (default: 12)
    --loss_target L         Stop a training call once a step's loss is below L
                            (default: 0.6; 0 or negative disables)
    --train-ratio RATIO     Fraction used for training (default: 0.8)
    --training-source S     "ground_truth" (fine-tune on the label) or
                            "self_generated" (train on the model's own revision)
    --revision-prompt P     Revision prompt preset for self_generated:
                            "default" or "fewshot"
    --think_mode M          "rationale" (default; "{rationale}\n</think>\n\n
                            {answer}" with a rationale that concludes the
                            answer, stop on the answer loss), "baseline"
                            (model's own reasoning in the unmasked prefix,
                            answer only in the loss), "empty"
                            ("</think>\n\n{answer}"), or "none" (old target)
    --[no]close_think       Deprecated alias: --noclose_think == --think_mode none
    --rehearsal_k K         Fold K self-distillation examples from correct
                            trained-split items into every correction step
                            (default 0)
    --rehearsal_max_tokens N  Skip rehearsal items whose baseline is longer than
                            N tokens (default 768)
    --rehearsal_weight W    Multiplier on the mean rehearsal gradient (default 1.0)
    --rehearsal_margin M    Rehearsal hinge: a rehearsal example's gradient counts
                            on a step only when its loss is more than M above its
                            loss at the call's first step (default 0.05)
    --rationale_max_tokens N  Cap the rationale in the target at N tokens, cut at
                            a sentence boundary (default 512)
    --[no]train_correct_items  Train train-split items whose baseline is already
                            correct (default False: skipped, re-inferred at the
                            end, regressions reported as interference)
    --verify_steps N        After a correction reaches the loss target or cap,
                            generate and judge; while wrong and under the cap,
                            train N more steps and check again (default 0 = off)
    --learning_rate LR      StatefulLLM learning rate (default: the model's)
    --lora_rank R           LoRA rank (default 32)
    --lora_layers N         Trailing layers converted to LoRA (default 24)
    --lora_scale S          LoRA scale (default 10.0)
    --repeats N             Runs per seed with identical shuffle (noise control)
    --holdout-every-checkpoint  Probe the holdout set at every checkpoint
    --subset N              Only use first N items (for quick tests)
    --category CAT          Filter to specific category
    --output PATH           Output path for results JSON
    --load-dataset PATH     Load dataset from JSON file instead of default
    --no-browser            Don't open browser for results

Examples:
    # Quick test with 3 seeds and 10 items
    python scripts/run_meta_experiment.py --seeds 42,123,456 --subset 10

    # Full experiment with 5 seeds
    python scripts/run_meta_experiment.py --seeds 42,123,456,789,1011

    # Filter to geography category
    python scripts/run_meta_experiment.py --category geography --seeds 42,123
"""

import json
import pathlib
import sys
import webbrowser
import os

# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from absl import app, flags

import vizible
import adaptible

# Access eval submodule through package
MetaLearningConfig = adaptible.eval.MetaLearningConfig
MetaLearningExperiment = adaptible.eval.MetaLearningExperiment
MetaLearningResult = adaptible.eval.MetaLearningResult
TRAINING_SOURCES = adaptible.eval.TRAINING_SOURCES
REVISION_PROMPTS = adaptible.revise.REVISION_PROMPTS
THINK_MODES = adaptible.revise.THINK_MODES
DEFAULT_RATIONALE_MAX_TOKENS = adaptible.revise.DEFAULT_RATIONALE_MAX_TOKENS
generate_default_dataset = adaptible.eval.generate_default_dataset
load_dataset = adaptible.eval.load_dataset
_harness = adaptible.eval.harness
DEFAULT_LORA_RANK = _harness.DEFAULT_LORA_RANK
DEFAULT_LORA_LAYERS = _harness.DEFAULT_LORA_LAYERS
DEFAULT_LORA_SCALE = _harness.DEFAULT_LORA_SCALE
lora_model_kwargs = _harness.lora_model_kwargs

_NAME = flags.DEFINE_string("name", "meta_experiment", "Experiment name")
_SEEDS = flags.DEFINE_string("seeds", "42,123,456", "Comma-separated random seeds")
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint_interval", 10, "Checkpoint every N training events"
)
_ITERATIONS = flags.DEFINE_integer(
    "iterations", 12, "Step cap per training call (exact count if --loss_target <= 0)"
)
_LOSS_TARGET = flags.DEFINE_float(
    "loss_target",
    0.6,
    "Stop each training call as soon as a step's loss is below this; "
    "--iterations is then a cap. The greedy answer flips to the correction at "
    "~0.6 with the reasoning intact; driving the loss to ~0 collapses the "
    "reasoning. 0 or a negative value disables the target and trains exactly "
    "--iterations steps.",
)
_TRAIN_RATIO = flags.DEFINE_float("train_ratio", 0.8, "Train/holdout split ratio")
_TRAINING_SOURCE = flags.DEFINE_enum(
    "training_source",
    "ground_truth",
    list(TRAINING_SOURCES),
    "What the model is trained on: the dataset label (ground_truth) or its own "
    "revision of its baseline answer (self_generated).",
)
_REVISION_PROMPT = flags.DEFINE_string(
    "revision_prompt",
    "default",
    f"Revision prompt preset used with --training_source self_generated; one of "
    f"{', '.join(REVISION_PROMPTS)}.",
)
_THINK_MODE = flags.DEFINE_enum(
    "think_mode",
    "rationale",
    list(THINK_MODES),
    "How the training target treats the chat template's open <think> tag: "
    "'rationale' (train on '{rationale}\\n</think>\\n\\n{revision}' with a "
    "rationale that concludes the revision; loss target on the answer tokens), "
    "'baseline' (model's own reasoning in the unmasked prefix, answer only in "
    "the loss), 'empty' (train on '</think>\\n\\n{revision}'), or 'none' (old "
    "malformed target).",
)
_CLOSE_THINK = flags.DEFINE_boolean(
    "close_think",
    None,
    "Deprecated; use --think_mode. --noclose_think is --think_mode none.",
)
_REHEARSAL_K = flags.DEFINE_integer(
    "rehearsal_k",
    0,
    "Fold K rehearsal examples into every correction step, from trained-split "
    "items the model already answered correctly (self-distillation): each step "
    "applies grad(correction) + rehearsal_weight * mean(grad(rehearsal)), one "
    "single-sequence pass per example, and stops on the correction loss alone. "
    "0 disables.",
)
_REHEARSAL_MAX_TOKENS = flags.DEFINE_integer(
    "rehearsal_max_tokens",
    768,
    "Exclude items whose baseline response is longer than this many tokens from "
    "the rehearsal pool.",
)
_REHEARSAL_WEIGHT = flags.DEFINE_float(
    "rehearsal_weight",
    1.0,
    "Multiplier on the mean rehearsal gradient in the joint step (>= 0).",
)
_REHEARSAL_MARGIN = flags.DEFINE_float(
    "rehearsal_margin",
    0.05,
    "Rehearsal hinge: a rehearsal example's gradient is applied on a step only "
    "when its loss is more than this above its loss at the call's first step "
    "(>= 0). Rehearsal anchors the model; it is never minimised on its own.",
)
_RATIONALE_MAX_TOKENS = flags.DEFINE_integer(
    "rationale_max_tokens",
    DEFAULT_RATIONALE_MAX_TOKENS,
    "Cap the rationale placed in the training target at this many tokens, cut "
    "at the last sentence boundary. Under --think_mode rationale an output "
    "with no </think> is taken whole as the rationale (the model reasons "
    "inside the open think block); an item with no rationale at all is skipped.",
)
_TRAIN_CORRECT_ITEMS = flags.DEFINE_boolean(
    "train_correct_items",
    False,
    "Train train-split items whose baseline answer is already judged correct. "
    "Off by default: such items are skipped (not trained, not in any window, "
    "not holdout), re-inferred at the end, and the ones that regressed are "
    "reported as interference. For self_generated this skip is an oracle.",
)
_VERIFY_STEPS = flags.DEFINE_integer(
    "verify_steps",
    0,
    "Verify-after-target: once a correction's training call returns (loss "
    "target reached or cap), generate the item's answer and judge it; while it "
    "is wrong and the step cap has not been reached, train this many more "
    "steps (no loss target) and check again. 0 disables.",
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", None, "StatefulLLM learning rate (default: the model's)."
)
_LORA_RANK = flags.DEFINE_integer(
    "lora_rank", DEFAULT_LORA_RANK, "LoRA rank of every converted linear layer."
)
_LORA_LAYERS = flags.DEFINE_integer(
    "lora_layers",
    DEFAULT_LORA_LAYERS,
    "Number of trailing transformer layers whose linears are converted to LoRA.",
)
_LORA_SCALE = flags.DEFINE_float(
    "lora_scale", DEFAULT_LORA_SCALE, "LoRA scale (alpha / rank)."
)
_REPEATS = flags.DEFINE_integer(
    "repeats", 1, "Runs per seed with an identical shuffle (noise control arm)"
)
_HOLDOUT_EVERY_CHECKPOINT = flags.DEFINE_boolean(
    "holdout_every_checkpoint", False, "Evaluate the holdout set at every checkpoint"
)
_SUBSET = flags.DEFINE_integer("subset", None, "Only use first N items")
_CATEGORY = flags.DEFINE_string("category", None, "Filter to specific category")
_OUTPUT = flags.DEFINE_string(
    "output", None, "Output path for results JSON (default: outputs/meta/<name>.json)"
)
_LOAD_DATASET = flags.DEFINE_string("load_dataset", None, "Load dataset from JSON file")
_NO_BROWSER = flags.DEFINE_boolean("no_browser", False, "Don't open browser")


def generate_summary_html(result: MetaLearningResult, output_path: pathlib.Path) -> str:
    """Generate a simple HTML summary of the meta-learning results."""
    html_path = output_path.with_suffix(".html")

    # Build trajectory table rows - sort by (meta_score or 0, accuracy, net_learning)
    def sort_key(item):
        seed, traj = item
        score = (
            traj.meta_learning_score if traj.meta_learning_score is not None else 0.0
        )
        return (score, traj.final_trained_accuracy, traj.total_net_learning)

    trajectory_rows = []
    for seed, traj in sorted(result.trajectories.items(), key=sort_key, reverse=True):
        is_best = seed == result.best_seed
        is_worst = seed == result.worst_seed
        badge = " 🏆" if is_best else " ⚠️" if is_worst else ""
        row_class = "best" if is_best else "worst" if is_worst else ""
        score_str = (
            f"{traj.meta_learning_score:.4f}"
            if traj.meta_learning_score is not None
            else f"N/A ({traj.meta_learning_score_reason})"
        )
        holdout_str = (
            f"{traj.holdout_accuracy:.1%}"
            if traj.holdout_accuracy is not None
            else "N/A"
        )

        trajectory_rows.append(f"""
            <tr class="{row_class}">
                <td>{seed}{badge}</td>
                <td>{score_str}</td>
                <td>{traj.final_trained_accuracy:.1%}</td>
                <td>{holdout_str}</td>
                <td>{traj.total_net_learning}</td>
                <td>{traj.window_sizes}</td>
                <td>{len(traj.checkpoints)}</td>
                <td>{traj.mean_train_steps:.1f} / {traj.train_cap_hits} cap</td>
                <td>{traj.skipped_correct_count} / {traj.skipped_correct_regressed}</td>
                <td>{f"{traj.verified_count}/{len(traj.verified)}" if result.config.verify_steps else "off"}</td>
                <td>{traj.total_time_seconds:.1f}s</td>
            </tr>
            """)

    # Build checkpoint progression for best seed
    checkpoint_data = []
    if result.best_seed is not None:
        best_traj = result.trajectories[result.best_seed]
        for cp in best_traj.checkpoints:
            checkpoint_data.append(
                {
                    "step": cp.step,
                    "accuracy": cp.post_accuracy,
                    "improvement_rate": cp.window_improvement_rate,
                    "forgetting_rate": cp.window_forgetting_rate,
                    "net_learning": cp.net_learning,
                }
            )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Meta-Learning Results: {result.config.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .config {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .config-item {{
            text-align: center;
        }}
        .config-item .label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .config-item .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .summary {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            text-align: center;
        }}
        .summary-item .label {{
            font-size: 14px;
            color: #888;
        }}
        .summary-item .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .summary-item .value.positive {{
            color: #4CAF50;
        }}
        .summary-item .value.negative {{
            color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        tr.best {{
            background: #e8f5e9;
        }}
        tr.worst {{
            background: #ffebee;
        }}
        .chart-container {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .interpretation {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #2196F3;
        }}
        .interpretation h3 {{
            margin-top: 0;
            color: #1976D2;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 12px;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Meta-Learning Experiment Results ({result.config.training_source})</h1>

    <div class="config" style="border-left: 6px solid {'#198754' if result.config.training_source == 'self_generated' else '#ffc107'};">
        <strong>Training source: <code>{result.config.training_source}</code></strong> &mdash;
        {"the model was trained on its own revisions (self-correction)." if result.config.training_source == "self_generated" else "the model was fine-tuned on the ground-truth label. This measures absorbing a supplied correction, <strong>not</strong> self-correction."}
    </div>

    <div class="config">
        <h2 style="margin-top: 0;">Configuration</h2>
        <div class="config-grid">
            <div class="config-item">
                <div class="label">Name</div>
                <div class="value" style="font-size: 16px;">{result.config.name}</div>
            </div>
            <div class="config-item">
                <div class="label">Training Source</div>
                <div class="value" style="font-size: 16px;">{result.config.training_source}</div>
            </div>
            <div class="config-item">
                <div class="label">Revision Prompt</div>
                <div class="value" style="font-size: 16px;">{result.config.revision_prompt}</div>
            </div>
            <div class="config-item">
                <div class="label">Think Mode</div>
                <div class="value" style="font-size: 16px;">{result.config.think_mode}</div>
            </div>
            <div class="config-item">
                <div class="label">Rehearsal k</div>
                <div class="value">{result.config.rehearsal_k}</div>
            </div>
            <div class="config-item">
                <div class="label">Rehearsal Max Tokens</div>
                <div class="value">{result.config.rehearsal_max_tokens}</div>
            </div>
            <div class="config-item">
                <div class="label">Rehearsal Weight</div>
                <div class="value">{result.config.rehearsal_weight:g}</div>
            </div>
            <div class="config-item">
                <div class="label">Rehearsal Margin</div>
                <div class="value">{result.config.rehearsal_margin:g}</div>
            </div>
            <div class="config-item">
                <div class="label">Rationale Max Tokens</div>
                <div class="value">{result.config.rationale_max_tokens}</div>
            </div>
            <div class="config-item">
                <div class="label">Train Correct Items</div>
                <div class="value" style="font-size: 16px;">{result.config.train_correct_items}</div>
            </div>
            <div class="config-item">
                <div class="label">Verify Steps</div>
                <div class="value">{result.config.verify_steps}</div>
            </div>
            <div class="config-item">
                <div class="label">LoRA (rank / layers / scale)</div>
                <div class="value" style="font-size: 16px;">{" / ".join(f"{v:g}" for v in _harness.lora_settings(result.model_kwargs))}</div>
            </div>
            <div class="config-item">
                <div class="label">Repeats / Seed</div>
                <div class="value">{result.config.repeats}</div>
            </div>
            <div class="config-item">
                <div class="label">Seeds</div>
                <div class="value" style="font-size: 18px;">{len(result.config.seeds)}</div>
            </div>
            <div class="config-item">
                <div class="label">Checkpoint Interval</div>
                <div class="value">{result.config.checkpoint_interval}</div>
            </div>
            <div class="config-item">
                <div class="label">Training Step Cap</div>
                <div class="value">{result.config.training_iterations}</div>
            </div>
            <div class="config-item">
                <div class="label">Loss Target</div>
                <div class="value">{result.config.loss_target if result.config.loss_target is not None else "off"}</div>
            </div>
            <div class="config-item">
                <div class="label">Train Ratio</div>
                <div class="value">{result.config.train_ratio:.0%}</div>
            </div>
            <div class="config-item">
                <div class="label">Dataset</div>
                <div class="value" style="font-size: 16px;">{result.dataset_name}</div>
            </div>
        </div>
    </div>

    <div class="summary">
        <h2 style="margin-top: 0;">Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="label">Best Seed</div>
                <div class="value positive">{result.best_seed or 'N/A'}</div>
            </div>
            <div class="summary-item">
                <div class="label">Worst Seed</div>
                <div class="value negative">{result.worst_seed or 'N/A'}</div>
            </div>
            <div class="summary-item">
                <div class="label">Score Variance (across seeds)</div>
                <div class="value">{f"{result.score_variance:.6f}" if result.score_variance is not None else "N/A"}</div>
            </div>
            <div class="summary-item">
                <div class="label">Within-Seed Variance (repeats)</div>
                <div class="value">{f"{result.within_seed_variance:.6f}" if result.within_seed_variance is not None else "N/A"}</div>
            </div>
            <div class="summary-item">
                <div class="label">Across-Seed Variance</div>
                <div class="value">{f"{result.across_seed_variance:.6f}" if result.across_seed_variance is not None else "N/A"}</div>
            </div>
            <div class="summary-item">
                <div class="label">Signal-to-Noise</div>
                <div class="value">{f"{result.signal_to_noise:.3f}" if result.signal_to_noise is not None else "N/A"}</div>
            </div>
        </div>
    </div>

    <h2>Trajectory Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Seed</th>
                <th>Meta-Learning Score</th>
                <th>Final Trained-Item Accuracy</th>
                <th>Holdout Accuracy</th>
                <th>Net Learning</th>
                <th>Window Sizes</th>
                <th>Checkpoints</th>
                <th>Mean Steps / Cap Hits</th>
                <th>Skipped Correct / Regressed</th>
                <th>Verified</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(trajectory_rows)}
        </tbody>
    </table>

    <div class="interpretation">
        <h3>Interpreting the Meta-Learning Score</h3>
        <p>The meta-learning score measures how learning efficiency changes over time:</p>
        <ul>
            <li><strong>Score &gt; 0:</strong> Model is "learning to learn" - improvements accelerate and forgetting decelerates</li>
            <li><strong>Score &lt; 0:</strong> Model is degrading - improvements slow down, forgetting increases</li>
            <li><strong>High variance:</strong> Meta-learning ability is sensitive to initialization</li>
        </ul>
        <p>Formula: <code>(late_improvement_rate - early_improvement_rate) + (early_forgetting_rate - late_forgetting_rate)</code>,
        using per-window rates (items trained since the previous checkpoint), averaged over the first and last third of checkpoints.
        A window with fewer than 5 trained items makes the score N/A.</p>
        <p>Chart rates below are the per-window rates, not cumulative.</p>
    </div>

    <h2>Learning Trajectory (Best Seed: {result.best_seed})</h2>
    <div class="chart-container">
        <canvas id="trajectoryChart"></canvas>
    </div>

    <script>
        const ctx = document.getElementById('trajectoryChart').getContext('2d');
        const checkpointData = {json.dumps(checkpoint_data)};

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: checkpointData.map(d => 'Step ' + d.step),
                datasets: [
                    {{
                        label: 'Trained-Item Accuracy (cumulative)',
                        data: checkpointData.map(d => d.accuracy * 100),
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        fill: true,
                        tension: 0.3,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Window Improvement Rate',
                        data: checkpointData.map(d => d.improvement_rate * 100),
                        borderColor: '#2196F3',
                        backgroundColor: 'transparent',
                        tension: 0.3,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Window Forgetting Rate',
                        data: checkpointData.map(d => d.forgetting_rate * 100),
                        borderColor: '#f44336',
                        backgroundColor: 'transparent',
                        tension: 0.3,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Net Learning',
                        data: checkpointData.map(d => d.net_learning),
                        borderColor: '#9C27B0',
                        backgroundColor: 'transparent',
                        tension: 0.3,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Rate (%)'
                        }},
                        min: 0,
                        max: 100
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Net Learning (count)'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }},
                    }}
                }}
            }}
        }});
    </script>

    <footer>
        Generated: {result.timestamp}<br>
        Results saved to: {output_path}
    </footer>
</body>
</html>
"""

    html_path.write_text(html)
    return str(html_path)


def main(_):
    # Parse seeds
    seeds = [int(s.strip()) for s in _SEEDS.value.split(",")]

    print()
    print("=" * 70)
    print("META-LEARNING EXPERIMENT")
    print("=" * 70)
    print()

    # Load or generate dataset
    if _LOAD_DATASET.value:
        print(f"Loading dataset from {_LOAD_DATASET.value}...")
        dataset = load_dataset(_LOAD_DATASET.value)
    else:
        print("Generating default dataset...")
        dataset = generate_default_dataset()

    print(f"Dataset: {dataset.name} ({len(dataset)} items)")
    print(f"Categories: {', '.join(sorted(dataset.categories))}")

    # Apply filters
    if _CATEGORY.value:
        dataset = dataset.by_category(_CATEGORY.value)
        print(f"Filtered to category '{_CATEGORY.value}': {len(dataset)} items")

    if _SUBSET.value:
        dataset = dataset.subset(list(range(min(_SUBSET.value, len(dataset)))))
        print(f"Using subset: {len(dataset)} items")

    if len(dataset) == 0:
        print("Error: No items in dataset after filtering")
        return 1

    # Create config
    config = MetaLearningConfig(
        name=_NAME.value,
        seeds=seeds,
        checkpoint_interval=_CHECKPOINT_INTERVAL.value,
        training_iterations=_ITERATIONS.value,
        loss_target=_LOSS_TARGET.value,
        train_ratio=_TRAIN_RATIO.value,
        training_source=_TRAINING_SOURCE.value,
        revision_prompt=_REVISION_PROMPT.value,
        think_mode=_THINK_MODE.value,
        close_think=_CLOSE_THINK.value,
        rehearsal_k=_REHEARSAL_K.value,
        rehearsal_max_tokens=_REHEARSAL_MAX_TOKENS.value,
        rehearsal_weight=_REHEARSAL_WEIGHT.value,
        rehearsal_margin=_REHEARSAL_MARGIN.value,
        rationale_max_tokens=_RATIONALE_MAX_TOKENS.value,
        repeats=_REPEATS.value,
        holdout_every_checkpoint=_HOLDOUT_EVERY_CHECKPOINT.value,
        train_correct_items=_TRAIN_CORRECT_ITEMS.value,
        verify_steps=_VERIFY_STEPS.value,
    )

    print()
    print("Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Training source: {config.training_source}")
    print(f"  Revision prompt: {config.revision_prompt}")
    print(f"  Think mode: {config.think_mode}")
    print(f"  Rehearsal k: {config.rehearsal_k}")
    print(f"  Rehearsal max tokens: {config.rehearsal_max_tokens}")
    print(f"  Rehearsal weight: {config.rehearsal_weight:g}")
    print(f"  Rehearsal margin: {config.rehearsal_margin:g}")
    print(f"  Rationale max tokens: {config.rationale_max_tokens}")
    print(f"  Train correct items: {config.train_correct_items}")
    print(f"  Verify steps: {config.verify_steps}")
    model_kwargs = lora_model_kwargs(
        rank=_LORA_RANK.value, layers=_LORA_LAYERS.value, scale=_LORA_SCALE.value
    )
    print(f"  {_harness.lora_settings_text(model_kwargs)}")
    if _LEARNING_RATE.value is not None:
        model_kwargs["learning_rate"] = _LEARNING_RATE.value
        print(f"  Learning rate: {_LEARNING_RATE.value}")
    print(f"  Seeds: {config.seeds}")
    print(f"  Repeats per seed: {config.repeats}")
    print(f"  Holdout every checkpoint: {config.holdout_every_checkpoint}")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    print(f"  Training step cap: {config.training_iterations}")
    print(f"  Loss target: {config.loss_target}")
    print(f"  Train ratio: {config.train_ratio}")
    print()

    # Determine output path
    if _OUTPUT.value:
        output_path = pathlib.Path(_OUTPUT.value)
    else:
        output_path = pathlib.Path(f"outputs/meta/{config.name}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run experiment
    experiment = MetaLearningExperiment(model_kwargs=model_kwargs)
    result = experiment.run(dataset, config, verbose=True)

    # Save results
    result.save(output_path)
    vizible.green(f"\nResults saved to: {output_path}")

    # Generate HTML report
    html_path = generate_summary_html(result, output_path)
    vizible.green(f"HTML report: file://{os.path.join(os.path.curdir, html_path)}")

    if not _NO_BROWSER.value:
        webbrowser.open_new_tab(f"file://{os.path.join(os.path.curdir, html_path)}")

    # Final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"  Training source: {config.training_source}")
    for label, seed in (
        ("Best seed", result.best_seed),
        ("Worst seed", result.worst_seed),
    ):
        print(f"  {label}:      {seed}")
        if seed is None:
            continue
        traj = result.trajectories[seed]
        score_str = (
            f"{traj.meta_learning_score:.4f}"
            if traj.meta_learning_score is not None
            else f"N/A ({traj.meta_learning_score_reason})"
        )
        holdout_str = (
            f"{traj.holdout_accuracy:.1%}"
            if traj.holdout_accuracy is not None
            else "N/A"
        )
        print(f"    Score:                  {score_str}")
        print(f"    Trained-item accuracy:  {traj.final_trained_accuracy:.1%}")
        print(f"    Holdout accuracy:       {holdout_str}")
        print(f"    Net learning:           {traj.total_net_learning}")
        print(f"    Window sizes:           {traj.window_sizes}")
        print()
    if result.score_variance is not None:
        print(f"  Score variance (across seeds): {result.score_variance:.6f}")
    else:
        print("  Score variance: N/A (insufficient scored seeds)")
    within, across, snr = (
        result.within_seed_variance,
        result.across_seed_variance,
        result.signal_to_noise,
    )
    print(
        "  Within-seed variance (repeats): "
        + (f"{within:.6f}" if within is not None else "N/A (use --repeats 2+)")
    )
    print(
        "  Across-seed variance:           "
        + (f"{across:.6f}" if across is not None else "N/A")
    )
    print(
        "  Signal-to-noise (across/within): "
        + (f"{snr:.3f}" if snr is not None else "N/A")
    )
    print()

    return 0


if __name__ == "__main__":
    app.run(main)
