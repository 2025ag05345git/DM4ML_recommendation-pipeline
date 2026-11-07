"""Validate CSV schemas, data quality, and interaction foreign keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def validate_frames(
    users: pd.DataFrame, movies: pd.DataFrame, interactions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "users": {"user_id", "age", "occupation", "preferred_genre"},
        "movies": {"item_id", "title", "genre", "release_year", "language", "language_code"},
        "interactions": {"interaction_id", "user_id", "item_id", "rating", "timestamp"},
    }
    for name, frame in (("users", users), ("movies", movies), ("interactions", interactions)):
        missing = required[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")

    source_issues: dict[str, dict[int, list[str]]] = {"users": {}, "movies": {}}
    for name, frame, key, numeric_columns in (
        ("users", users, "user_id", ("age",)),
        ("movies", movies, "item_id", ("release_year",)),
    ):
        issues = source_issues[name]

        def flag_source(mask: pd.Series, reason: str) -> None:
            for index in frame.index[mask.fillna(False)]:
                issues.setdefault(int(index), []).append(reason)

        flag_source(frame[key].isna(), f"missing_{key}")
        flag_source(frame[key].notna() & frame[key].duplicated(keep=False), f"duplicate_{key}")
        required_text = (
            ("occupation", "preferred_genre")
            if name == "users"
            else ("title", "genre", "language", "language_code")
        )
        for column in required_text:
            flag_source(frame[column].isna(), f"missing_{column}")
            wrong_type = frame[column].notna() & ~frame[column].map(lambda value: isinstance(value, str))
            flag_source(wrong_type, f"invalid_{column}_type")
        for column in numeric_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            flag_source(frame[column].notna() & numeric.isna(), f"invalid_{column}_type")
        if name == "movies":
            flag_source(frame["language"].astype(str).str.lower() != "english", "non_english_language")
            flag_source(frame["language_code"].astype(str).str.lower() != "en", "invalid_language_code")

    valid_users = users.drop(index=sorted(source_issues["users"])).copy()
    valid_movies = movies.drop(index=sorted(source_issues["movies"])).copy()
    interaction_issues: dict[int, list[str]] = {}

    def flag_interaction(mask: pd.Series, reason: str) -> None:
        for index in interactions.index[mask.fillna(False)]:
            interaction_issues.setdefault(int(index), []).append(reason)

    for column in ("interaction_id", "user_id", "item_id", "rating", "timestamp"):
        flag_interaction(interactions[column].isna(), f"missing_{column}")
    flag_interaction(
        interactions["interaction_id"].notna()
        & interactions["interaction_id"].duplicated(keep="first"),
        "duplicate_interaction_id",
    )
    ratings = pd.to_numeric(interactions["rating"], errors="coerce")
    flag_interaction(interactions["rating"].notna() & ratings.isna(), "invalid_rating_type")
    flag_interaction(ratings.notna() & ~ratings.between(1, 5), "rating_out_of_range")
    times = pd.to_datetime(interactions["timestamp"], errors="coerce", utc=True)
    flag_interaction(interactions["timestamp"].notna() & times.isna(), "invalid_timestamp")
    flag_interaction(
        interactions["user_id"].notna()
        & ~interactions["user_id"].astype(str).isin(set(valid_users["user_id"].astype(str))),
        "unknown_user",
    )
    flag_interaction(
        interactions["item_id"].notna()
        & ~interactions["item_id"].astype(str).isin(set(valid_movies["item_id"].astype(str))),
        "unknown_item",
    )
    invalid_indices = sorted(interaction_issues)
    invalid = interactions.loc[invalid_indices].copy() if invalid_indices else interactions.iloc[:0].copy()
    invalid["validation_errors"] = [";".join(interaction_issues[index]) for index in invalid_indices]
    valid = interactions.drop(index=invalid_indices).copy()
    valid["rating"] = pd.to_numeric(valid["rating"], errors="raise").astype(float)
    valid["timestamp"] = pd.to_datetime(valid["timestamp"], errors="raise", utc=True)

    all_issues = [
        *interaction_issues.values(),
        *source_issues["users"].values(),
        *source_issues["movies"].values(),
    ]
    report: dict[str, Any] = {
        "status": "passed_with_quarantine" if all_issues else "passed",
        "language_policy": "English only (language=English, language_code=en)",
        "row_counts": {
            "users_input": len(users), "users_valid": len(valid_users),
            "users_quarantined": len(source_issues["users"]),
            "movies_input": len(movies), "movies_valid": len(valid_movies),
            "movies_quarantined": len(source_issues["movies"]),
            "interactions_input": len(interactions), "interactions_valid": len(valid),
            "interactions_quarantined": len(invalid),
        },
        "missing_values": {
            name: {column: int(count) for column, count in frame.isna().sum().items()}
            for name, frame in (("users", users), ("movies", movies), ("interactions", interactions))
        },
        "quarantine_reasons": {
            reason: sum(reason in issues for issues in all_issues)
            for reason in sorted({reason for issues in all_issues for reason in issues})
        },
        "invalid_source_rows": {
            name: [
                {"row_index": index, "validation_errors": ";".join(issues[index])}
                for index in sorted(issues)
            ]
            for name, issues in source_issues.items()
        },
    }
    return invalid, report, valid, valid_users, valid_movies


def validate_files() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = ROOT / "data" / "raw"
    users = pd.read_csv(raw / "users.csv", dtype={"user_id": "string"})
    movies = pd.read_csv(raw / "movies.csv", dtype={"item_id": "string"})
    interactions = pd.read_csv(
        raw / "interactions.csv",
        dtype={"interaction_id": "string", "user_id": "string", "item_id": "string"},
    )
    invalid, report, valid, valid_users, valid_movies = validate_frames(users, movies, interactions)
    error_dir = ROOT / "data" / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(error_dir / "invalid_interactions.csv", index=False)
    for name, frame in (("users", users), ("movies", movies)):
        rows = report["invalid_source_rows"][name]
        bad = frame.loc[[row["row_index"] for row in rows]].copy() if rows else frame.iloc[:0].copy()
        bad["validation_errors"] = [row["validation_errors"] for row in rows]
        bad.to_csv(error_dir / f"invalid_{name}.csv", index=False)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log_lines = [
        "Validation result: " + report["status"],
        *(f"{name}: {count}" for name, count in report["row_counts"].items()),
        "Quarantine reasons:",
        *(f"  {name}: {count}" for name, count in report["quarantine_reasons"].items()),
    ]
    (reports / "validation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return invalid, report, valid, valid_users, valid_movies
