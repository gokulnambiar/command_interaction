from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def evaluate_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for approach in sorted({record["approach"] for record in records}):
        approach_records = [record for record in records if record["approach"] == approach]
        metrics[approach] = _metrics_for_groups(approach_records)
    return metrics


def save_outputs(
    output_dir: Path,
    evaluated_records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    sample_columns = [
        "instruction_type",
        "instruction",
        "approach",
        "prediction",
        "ground_truth",
        "exact_match",
        "stepwise_accuracy",
    ]
    sample_df = pd.DataFrame(evaluated_records)[sample_columns]
    sample_df.to_json(output_dir / "sample_predictions.json", orient="records", indent=2)

    summary_rows = []
    for approach, groups in metrics.items():
        for group_name, values in groups.items():
            summary_rows.append(
                {
                    "approach": approach,
                    "group": group_name,
                    "exact_match_accuracy": values["exact_match_accuracy"],
                    "stepwise_accuracy": values["stepwise_accuracy"],
                    "count": values["count"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "performance_breakdown.csv", index=False)
    _plot_metrics(summary_df, output_dir / "accuracy_comparison.png")


def sequence_exact_match(predicted: list[dict[str, str]], truth: list[dict[str, str]]) -> bool:
    return _normalize_sequence(predicted) == _normalize_sequence(truth)


def sequence_stepwise_accuracy(predicted: list[dict[str, str]], truth: list[dict[str, str]]) -> float:
    predicted_steps = _normalize_sequence(predicted)
    truth_steps = _normalize_sequence(truth)
    denominator = max(len(predicted_steps), len(truth_steps), 1)
    matches = sum(1 for pred, gold in zip(predicted_steps, truth_steps) if pred == gold)
    return matches / denominator


def _metrics_for_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped["overall"].append(record)
        grouped[record["instruction_type"]].append(record)

    metrics = {}
    for group_name, group_records in grouped.items():
        exact_scores = [float(record["exact_match"]) for record in group_records]
        step_scores = [record["stepwise_accuracy"] for record in group_records]
        metrics[group_name] = {
            "count": len(group_records),
            "exact_match_accuracy": round(sum(exact_scores) / len(exact_scores), 4),
            "stepwise_accuracy": round(sum(step_scores) / len(step_scores), 4),
        }
    return metrics


def _normalize_sequence(sequence: list[dict[str, str]]) -> list[tuple[str, str]]:
    normalized = []
    for step in sequence:
        normalized.append(
            (
                str(step.get("action", "")).strip().lower(),
                str(step.get("object", "")).strip().lower(),
            )
        )
    return normalized


def _plot_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    filtered = summary_df[summary_df["group"].isin(["overall", "simple", "multi_step"])].copy()
    if filtered.empty:
        return

    pivot = filtered.pivot(index="group", columns="approach", values="exact_match_accuracy").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(8, 4), rot=0)
    ax.set_ylabel("Exact match accuracy")
    ax.set_xlabel("")
    ax.set_title("Instruction-to-action accuracy")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)

