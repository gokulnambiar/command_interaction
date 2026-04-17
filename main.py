from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from src.data_utils import DEFAULT_SAMPLE_SIZES, ensure_alfred_subset
from src.evaluation import (
    evaluate_predictions,
    save_outputs,
    sequence_exact_match,
    sequence_stepwise_accuracy,
)
from src.parsers import RetrievalParser, RuleBasedParser


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "outputs"

    records, data_summary = ensure_alfred_subset(
        data_dir=data_dir,
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        seed=7,
    )

    train_records = [record for record in records if record["split"] == "train"]
    eval_records = [record for record in records if record["split"] != "train"]

    baseline_parser = RuleBasedParser().fit(train_records)
    retrieval_parser = RetrievalParser().fit(train_records)

    evaluated_rows = []
    for record in eval_records:
        baseline_prediction = baseline_parser.predict(record["instruction"])
        retrieval_prediction = retrieval_parser.predict(
            instruction=record["instruction"],
            instruction_type=record["instruction_type"],
        )
        evaluated_rows.extend(
            [
                _build_result_row(record, "baseline", baseline_prediction),
                _build_result_row(record, "retrieval", retrieval_prediction),
            ]
        )

    metrics = evaluate_predictions(evaluated_rows)
    save_outputs(output_dir=output_dir, evaluated_records=evaluated_rows, metrics=metrics)
    (output_dir / "dataset_summary.json").write_text(json.dumps(data_summary, indent=2))

    print("Saved outputs:")
    print(f"- {output_dir / 'metrics.json'}")
    print(f"- {output_dir / 'sample_predictions.json'}")
    print(f"- {output_dir / 'performance_breakdown.csv'}")
    print(f"- {output_dir / 'accuracy_comparison.png'}")
    print(f"- {output_dir / 'dataset_summary.json'}")


def _build_result_row(record: dict, approach: str, prediction: list[dict[str, str]]) -> dict:
    ground_truth = record["target_actions"]
    return {
        "record_id": record["record_id"],
        "split": record["split"],
        "instruction_type": record["instruction_type"],
        "instruction": record["instruction"],
        "approach": approach,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "exact_match": sequence_exact_match(prediction, ground_truth),
        "stepwise_accuracy": sequence_stepwise_accuracy(prediction, ground_truth),
    }


if __name__ == "__main__":
    main()
