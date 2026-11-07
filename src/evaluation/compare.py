"""Compare recommendation models and persist metrics and recommendations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate
from src.training.recommenders import build_recommendations

ROOT = Path(__file__).resolve().parents[2]


def compare_models(models, train, test, movies, k, relevance_threshold):
    genres = dict(zip(movies["item_id"].astype(str), movies["genre"].astype(str), strict=True))
    rows, all_recommendations = [], {}
    for name, model in models.items():
        recs = build_recommendations(model, test, train, k)
        all_recommendations[name] = recs
        rows.append(
            {"model": name, **evaluate(recs, test, genres, len(movies), k, relevance_threshold)}
        )
    return pd.DataFrame(rows), all_recommendations


def write_results(results, recommendations, relevance_threshold, holdout_count):
    reports = ROOT / "reports"
    models_dir = ROOT / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "model_evaluation.csv"
    markdown_path = reports / "model_evaluation.md"
    rec_path = models_dir / "recommendations.json"
    results.to_csv(csv_path, index=False, float_format="%.4f")
    metric_k = next(column for column in results.columns if column.startswith("precision_at_"))
    k = metric_k.rsplit("_", 1)[-1]
    lines = [
        "# Model evaluation results",
        "",
        f"At least the latest {holdout_count} event(s) per user are held out; ratings >= "
        f"{relevance_threshold:g} are relevant. K={k}.",
        "",
        f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | Coverage | Diversity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        values = row._asdict()
        lines.append(
            f"| {row.model} | {values[f'precision_at_{k}']:.4f} | "
            f"{values[f'recall_at_{k}']:.4f} | {values[f'ndcg_at_{k}']:.4f} | "
            f"{row.coverage:.4f} | {row.diversity:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec_path.write_text(json.dumps(recommendations, indent=2) + "\n", encoding="utf-8")
    return {"evaluation_csv": csv_path, "evaluation_markdown": markdown_path, "recommendations": rec_path}
