"""Build reusable user, movie, and interaction feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build_features(
    users: pd.DataFrame, movies: pd.DataFrame, train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff = train["timestamp"].max()
    user_stats = train.groupby("user_id").agg(
        user_activity_count=("interaction_id", "count"),
        user_average_rating=("rating", "mean"),
        user_last_interaction=("timestamp", "max"),
    )
    categories = train.merge(movies[["item_id", "genre"]], on="item_id", how="left")
    favorite = categories.groupby("user_id")["genre"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"
    ).rename("user_preferred_genre")
    user_features = users.merge(user_stats, on="user_id", how="left").merge(favorite, on="user_id", how="left")
    user_features["user_activity_count"] = user_features["user_activity_count"].fillna(0).astype(int)
    user_features["user_average_rating"] = user_features["user_average_rating"].fillna(0.0)
    user_features["days_since_last_interaction"] = (
        cutoff - pd.to_datetime(user_features["user_last_interaction"], utc=True)
    ).dt.days.fillna(-1).astype(int)
    user_features["user_preferred_genre"] = user_features["user_preferred_genre"].fillna("Unknown")

    movie_stats = train.groupby("item_id").agg(
        item_popularity=("interaction_id", "count"),
        item_average_rating=("rating", "mean"),
        item_last_interaction=("timestamp", "max"),
    )
    movie_features = movies.merge(movie_stats, on="item_id", how="left")
    movie_features["item_popularity"] = movie_features["item_popularity"].fillna(0).astype(int)
    movie_features["item_average_rating"] = movie_features["item_average_rating"].fillna(0.0)
    movie_features["days_since_last_interaction"] = (
        cutoff - pd.to_datetime(movie_features["item_last_interaction"], utc=True)
    ).dt.days.fillna(-1).astype(int)

    interaction_features = train.merge(
        users[["user_id", "age", "occupation"]], on="user_id", how="left"
    ).merge(movies[["item_id", "genre", "release_year", "language"]], on="item_id", how="left")
    interaction_features["days_since_last_interaction"] = (
        cutoff - interaction_features["timestamp"]
    ).dt.days
    return user_features, movie_features, interaction_features


def write_features(
    users: pd.DataFrame, movies: pd.DataFrame, interactions: pd.DataFrame
) -> None:
    output = ROOT / "data" / "features"
    output.mkdir(parents=True, exist_ok=True)
    users.to_csv(output / "users.csv", index=False)
    movies.to_csv(output / "movies.csv", index=False)
    interactions.to_csv(output / "interactions.csv", index=False)
