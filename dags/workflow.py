"""Dependency-checked scheduled workflow entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import TASK_NAMES, run_pipeline

DEPENDENCIES = {
    "ingestion": (),
    "validation": ("ingestion",),
    "preparation": ("validation",),
    "transformation": ("preparation",),
    "feature_management": ("preparation",),
    "model_training": ("transformation", "feature_management"),
    "model_evaluation": ("model_training", "preparation"),
    "reporting": ("model_evaluation", "validation"),
}


def validate_workflow_order() -> None:
    completed = set()
    for task in TASK_NAMES:
        missing = set(DEPENDENCIES[task]) - completed
        if missing:
            raise RuntimeError(f"Workflow dependency violation for {task}: {sorted(missing)}")
        completed.add(task)


if __name__ == "__main__":
    validate_workflow_order()
    run_pipeline()
