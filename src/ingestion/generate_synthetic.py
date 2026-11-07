"""Generate deterministic English-language movie, user, and rating data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GENRES = (
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "Horror", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
)
ADJECTIVES = (
    "Bright", "Hidden", "Quiet", "Golden", "Distant", "Silver",
    "Endless", "Forgotten", "Gentle", "Restless", "Northern", "Secret",
)
NOUNS = (
    "Harbor", "Garden", "Journey", "Promise", "Mountain", "Lantern",
    "River", "Portrait", "Voyage", "Station", "Valley", "Story",
)


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or ROOT / "configs" / "default.json").open(encoding="utf-8") as file:
        return json.load(file)


def generate(
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or load_config()
    rng = np.random.default_rng(int(config["random_seed"]))
    movie_count, user_count = int(config["movies"]), int(config["users"])
    per_user = int(config["interactions_per_user"])
    featured_per_genre = int(config["featured_movies_per_genre"])
    if movie_count < 2 or user_count < 1 or per_user < featured_per_genre + 1 or per_user >= movie_count:
        raise ValueError(
            "Require movies >= 2, users >= 1, and interactions_per_user >= "
            "featured_movies_per_genre + 1"
        )
    if featured_per_genre < 2 or movie_count < len(GENRES) * featured_per_genre:
        raise ValueError("Movie catalog must contain at least two featured movies per genre")

    genre_indices = np.arange(movie_count) % len(GENRES)
    year_values = rng.integers(1920, 2026, size=movie_count)
    movies = pd.DataFrame(
        {
            "item_id": [f"m{index:05d}" for index in range(movie_count)],
            "title": [
                f"The {ADJECTIVES[index % len(ADJECTIVES)]} "
                f"{NOUNS[(index // len(ADJECTIVES)) % len(NOUNS)]} "
                f"{GENRES[genre_indices[index]]} {index + 1:05d}"
                for index in range(movie_count)
            ],
            "genre": [GENRES[index] for index in genre_indices],
            "release_year": year_values,
            "language": [str(config.get("language", "English"))] * movie_count,
            "language_code": ["en"] * movie_count,
            "overview": [
                f"A fictional {GENRES[genre_indices[index]].lower()} story set in "
                f"{int(year_values[index])}, created for this synthetic dataset."
                for index in range(movie_count)
            ],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [f"u{index:04d}" for index in range(user_count)],
            "age": [np.nan if index % 97 == 0 else int(rng.integers(18, 76)) for index in range(user_count)],
            "occupation": rng.choice(
                ["student", "engineer", "teacher", "artist", "healthcare", "retired"],
                size=user_count,
            ),
            "preferred_genre": [GENRES[index % len(GENRES)] for index in range(user_count)],
        }
    )

    genres_by_movie = movies["genre"].to_numpy()
    ids_by_movie = movies["item_id"].to_numpy()
    featured_indices = {
        genre_index + len(GENRES) * slot
        for genre_index in range(len(GENRES))
        for slot in range(featured_per_genre)
    }
    ordinary_indices = np.array(
        [index for index in range(movie_count) if index not in featured_indices]
    )
    start = pd.Timestamp("2023-01-01", tz="UTC")
    rows: list[dict[str, Any]] = []
    for user_index in range(user_count):
        favorite = users.iloc[user_index]["preferred_genre"]
        genre_index = GENRES.index(favorite)
        test_slot = user_index % featured_per_genre
        test_movie_index = genre_index + len(GENRES) * test_slot
        support_movie_indices = [
            genre_index + len(GENRES) * slot
            for slot in range(featured_per_genre)
            if slot != test_slot
        ]
        eligible = ordinary_indices
        favorite_pool = eligible[genres_by_movie[eligible] == favorite]
        other_pool = eligible[genres_by_movie[eligible] != favorite]
        ordinary_count = per_user - featured_per_genre
        favorite_count = min(int(ordinary_count * 0.7), ordinary_count)
        random_movies = np.concatenate(
            [
                rng.choice(favorite_pool, size=favorite_count, replace=False),
                rng.choice(other_pool, size=ordinary_count - favorite_count, replace=False),
            ]
        )
        chosen = np.concatenate([random_movies, support_movie_indices])
        chosen = chosen[rng.permutation(len(chosen))]
        days = np.sort(rng.integers(0, 1200, size=len(chosen)))
        for sequence, (movie_index, day) in enumerate(zip(chosen, days, strict=True)):
            matched = genres_by_movie[movie_index] == favorite
            rating = int(np.clip(round(rng.normal(4.1 if matched else 3.0, 0.9)), 1, 5))
            rows.append(
                {
                    "interaction_id": f"i{user_index:04d}_{sequence:03d}",
                    "user_id": users.iloc[user_index]["user_id"],
                    "item_id": ids_by_movie[movie_index],
                    "rating": rating,
                    "timestamp": (start + pd.Timedelta(days=int(day))).isoformat(),
                }
            )
        rows.append(
            {
                "interaction_id": f"i{user_index:04d}_{per_user - 1:03d}",
                "user_id": users.iloc[user_index]["user_id"],
                "item_id": ids_by_movie[test_movie_index],
                "rating": 5,
                "timestamp": (start + pd.Timedelta(days=1500 + user_index)).isoformat(),
            }
        )
    interactions = pd.DataFrame(rows)

    sample = interactions.iloc[0].copy()
    defects = []
    duplicate = sample.copy()
    defects.append(duplicate)
    bad_rating = sample.copy()
    bad_rating["interaction_id"], bad_rating["rating"] = "bad_rating", 6
    defects.append(bad_rating)
    bad_timestamp = sample.copy()
    bad_timestamp["interaction_id"], bad_timestamp["timestamp"] = "bad_timestamp", "not-a-date"
    defects.append(bad_timestamp)
    unknown_user = sample.copy()
    unknown_user["interaction_id"], unknown_user["user_id"] = "unknown_user", "u_missing"
    defects.append(unknown_user)
    missing_rating = sample.copy()
    missing_rating["interaction_id"], missing_rating["rating"] = "missing_rating", np.nan
    defects.append(missing_rating)
    interactions = pd.concat([interactions, pd.DataFrame(defects)], ignore_index=True)
    return users, movies, interactions


def write_raw(config: dict[str, Any] | None = None) -> dict[str, Path]:
    users, movies, interactions = generate(config)
    output = ROOT / "data" / "raw"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "users": output / "users.csv",
        "movies": output / "movies.csv",
        "interactions": output / "interactions.csv",
    }
    users.to_csv(paths["users"], index=False)
    movies.to_csv(paths["movies"], index=False)
    interactions.to_csv(paths["interactions"], index=False)
    return paths


if __name__ == "__main__":
    for name, path in write_raw().items():
        print(f"{name}: {path.relative_to(ROOT)}")
