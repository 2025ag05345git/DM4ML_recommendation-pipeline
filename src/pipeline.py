"""Run ingestion through reporting as one reproducible pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.compare import compare_models, write_results
from src.features.build_features import build_features, write_features
from src.ingestion.generate_synthetic import load_config, write_raw
from src.preparation.prepare import prepare, write_prepared
from src.reporting.build_report import build_final_report
from src.training.recommenders import ContentBasedRecommender, PopularityRecommender
from src.transformation.transform import build_derived, write_derived
from src.validation.validate import validate_files

ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = (
    "ingestion", "validation", "preparation", "transformation",
    "feature_management", "model_training", "model_evaluation", "reporting",
)


def run_pipeline() -> dict:
    config = load_config()
    if config.get("storage_backend") != "csv":
        raise ValueError("This version supports storage_backend=csv; PostgreSQL is a future placeholder")
    if int(config["top_k"]) not in (5, 10):
        raise ValueError("Assignment evaluation requires top_k to be 5 or 10")
    if not 1 <= float(config["relevance_threshold"]) <= 5:
        raise ValueError("relevance_threshold must be between 1 and 5")
    started = datetime.now(timezone.utc)
    log = ["Movie recommendation pipeline", f"Started UTC: {started.isoformat()}"]
    for task in TASK_NAMES:
        log.append(f"START {task}")
        if task == "ingestion":
            raw_paths = write_raw(config)
            users = pd.read_csv(raw_paths["users"], dtype={"user_id": "string"})
            movies = pd.read_csv(raw_paths["movies"], dtype={"item_id": "string"})
        elif task == "validation":
            _, validation_report, interactions, users, movies = validate_files()
        elif task == "preparation":
            users, movies, train, test = prepare(
                users, movies, interactions, int(config["test_interactions_per_user"])
            )
            write_prepared(users, movies, train, test)
        elif task == "transformation":
            matrix, activity, popularity_table = build_derived(train, movies)
            write_derived(matrix, activity, popularity_table)
        elif task == "feature_management":
            user_features, movie_features, interaction_features = build_features(users, movies, train)
            write_features(user_features, movie_features, interaction_features)
        elif task == "model_training":
            movie_ids = movies["item_id"].astype(str).tolist()
            models = {
                "popularity": PopularityRecommender().fit(train, movie_ids),
                "content_based_genre": ContentBasedRecommender().fit(train, movies),
            }
        elif task == "model_evaluation":
            results, recommendations = compare_models(
                models, train, test, movies, int(config["top_k"]),
                float(config["relevance_threshold"]),
            )
            result_paths = write_results(
                results, recommendations, float(config["relevance_threshold"]),
                int(config["test_interactions_per_user"]),
            )
        elif task == "reporting":
            report_paths = build_final_report(results, config, validation_report["row_counts"])
            experiment = {
                "run_id": started.strftime("%Y%m%dT%H%M%SZ"),
                "started_at_utc": started.isoformat(),
                "parameters": {
                    "random_seed": config["random_seed"],
                    "users": config["users"],
                    "movies": config["movies"],
                    "interactions_per_user": config["interactions_per_user"],
                    "featured_movies_per_genre": config["featured_movies_per_genre"],
                    "test_interactions_per_user": config["test_interactions_per_user"],
                    "top_k": config["top_k"],
                    "relevance_threshold": config["relevance_threshold"],
                    "personalized_model": "rating-weighted genre content model",
                },
                "metrics": {
                    row.model: {
                        key: int(value) if key == "evaluated_users" else float(value)
                        for key, value in row._asdict().items() if key != "model"
                    }
                    for row in results.itertuples(index=False)
                },
                "artifacts": {
                    "evaluation_csv": str(result_paths["evaluation_csv"].relative_to(ROOT)),
                    "recommendations": str(result_paths["recommendations"].relative_to(ROOT)),
                    "validation_report": "reports/validation_report.json",
                    "final_report_markdown": str(report_paths[0].relative_to(ROOT)),
                    "final_report_pdf": str(report_paths[1].relative_to(ROOT)),
                },
            }
            model_dir = ROOT / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "experiment_run.json").write_text(
                json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
            )
        log.append(f"COMPLETE {task}")
    log.extend([f"Finished UTC: {datetime.now(timezone.utc).isoformat()}", "STATUS: SUCCESS"])
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "workflow_run.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"Movies in catalog: {validation_report['row_counts']['movies_valid']}")
    print(results.to_string(index=False))
    return {"results": results, "validation": validation_report, "experiment": experiment}


if __name__ == "__main__":
    run_pipeline()
