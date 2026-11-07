"""Clean source records and perform a per-user chronological holdout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def prepare(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    interactions: pd.DataFrame,
    test_interactions_per_user: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_interactions_per_user < 1:
        raise ValueError("test_interactions_per_user must be at least one")
    users, movies, interactions = users.copy(), movies.copy(), interactions.copy()
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    median_age = users["age"].median()
    users["age"] = users["age"].fillna(median_age if pd.notna(median_age) else 0).astype(int)
    users["occupation"] = users["occupation"].fillna("unknown").astype(str).str.strip().str.lower()
    movies["title"] = movies["title"].fillna("Untitled").astype(str).str.strip()
    movies["genre"] = movies["genre"].fillna("Unknown").astype(str).str.strip().str.title()
    movies["language"] = movies["language"].fillna("English").astype(str).str.strip()
    movies["release_year"] = pd.to_numeric(movies["release_year"], errors="coerce").astype("Int64")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], utc=True, errors="raise")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="raise").astype(float)
    interactions = interactions.drop_duplicates("interaction_id", keep="first")

    ordered = interactions.sort_values(["user_id", "timestamp", "interaction_id"])
    if (ordered.groupby("user_id").size() <= test_interactions_per_user).any():
        raise ValueError("Each user must have more valid interactions than the test holdout size")
    tail = ordered.groupby("user_id", sort=False).tail(test_interactions_per_user)
    cutoff = tail.groupby("user_id")["timestamp"].min()
    test_mask = ordered["timestamp"] >= ordered["user_id"].map(cutoff)
    test = ordered.loc[test_mask].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    train = ordered.loc[~test_mask].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    if train["user_id"].nunique() != test["user_id"].nunique():
        raise ValueError("Temporal split failed to retain train and test data for each user")
    return users, movies, train, test


def write_prepared(
    users: pd.DataFrame, movies: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> None:
    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    users.to_csv(output / "users_clean.csv", index=False)
    movies.to_csv(output / "movies_clean.csv", index=False)
    train.to_csv(output / "train_interactions.csv", index=False)
    test.to_csv(output / "test_interactions.csv", index=False)
