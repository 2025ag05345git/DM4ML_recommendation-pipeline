"""Create a user-item matrix, user activity summaries, and movie popularity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build_derived(
    train: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    activity = train.groupby("user_id").agg(
        activity_count=("interaction_id", "count"),
        average_rating=("rating", "mean"),
        last_interaction=("timestamp", "max"),
    ).reset_index()
    popularity = train.groupby("item_id").agg(
        interaction_count=("interaction_id", "count"),
        average_rating=("rating", "mean"),
    ).reset_index().merge(movies[["item_id", "title", "genre"]], on="item_id", how="right")
    popularity["interaction_count"] = popularity["interaction_count"].fillna(0).astype(int)
    popularity["average_rating"] = popularity["average_rating"].fillna(0.0)
    return matrix, activity, popularity


def write_derived(
    matrix: pd.DataFrame, activity: pd.DataFrame, popularity: pd.DataFrame
) -> None:
    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    matrix.rename_axis("user_id").to_csv(output / "user_item_matrix.csv")
    activity.to_csv(output / "user_activity_summary.csv", index=False)
    popularity.to_csv(output / "item_popularity.csv", index=False)
