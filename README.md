# command_interaction

This project turns ALFRED language instructions into structured action sequences without using images or the simulator. It keeps the setup small enough to run locally and simple enough to explain in an interview.

## Problem

Given a natural language household instruction, predict a JSON action plan:

```json
[
  {"action": "PickupObject", "object": "mug"},
  {"action": "PutObject", "object": "mug -> coffeemachine"}
]
```

## Dataset

The pipeline downloads the official ALFRED lite annotation archive and builds a manageable subset from trajectory JSON files only. It extracts:

- goal instructions from `task_desc`
- step instructions from `high_descs`
- aligned expert actions from `plan.high_pddl`

Single high-level steps are treated as `simple` instructions. Goal-level plans and grouped subgoals are treated as `multi_step`.

## Approach

Two parsers are evaluated on the same subset:

- `baseline`: rule-based parsing with verb heuristics and object matching from the training vocabulary
- `retrieval`: TF-IDF nearest-neighbor retrieval that returns the action sequence from the most similar training instruction

Both methods emit the same JSON schema:

```json
[
  {"action": "...", "object": "..."}
]
```

## Evaluation

The project reports:

- exact match accuracy
- step-wise accuracy
- separate results for `simple` and `multi_step` instructions

Saved outputs:

- `outputs/metrics.json`
- `outputs/sample_predictions.json`
- `outputs/performance_breakdown.csv`
- `outputs/accuracy_comparison.png`
- `outputs/dataset_summary.json`

## Run

```bash
cd /Users/gokulnambiar/Codex/command_interaction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

The first run downloads the ALFRED lite archive into `data/raw/`, extracts it, builds the subset, runs both parsers, and writes results into `outputs/`.
