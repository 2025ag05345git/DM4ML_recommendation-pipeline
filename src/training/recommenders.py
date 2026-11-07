"""Popularity baseline and personalized genre-based content recommender."""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    def __init__(self) -> None:
        self.ranking: list[str] = []

    def fit(self, train: pd.DataFrame, movie_ids: list[str]) -> "PopularityRecommender":
        stats = train.groupby("item_id").agg(
            count=("interaction_id", "count"), mean_rating=("rating", "mean")
        )
        stats["score"] = stats["count"] * stats["mean_rating"]
        ranked = stats.sort_values(["score", "count"], ascending=False).index.astype(str).tolist()
        self.ranking = ranked + [movie for movie in movie_ids if movie not in stats.index]
        return self

    def recommend(self, user_id: str, seen: set[str], k: int) -> list[str]:
        del user_id
        return [movie for movie in self.ranking if movie not in seen][:k]


class ContentBasedRecommender:
    """Rank movies by a user's rating-weighted genre preference."""

    def __init__(self) -> None:
        self.movie_ids: list[str] = []
        self.movie_genres: dict[str, str] = {}
        self.user_genre_scores: dict[str, dict[str, float]] = {}
        self.popularity: PopularityRecommender | None = None

    def fit(
        self, train: pd.DataFrame, movies: pd.DataFrame
    ) -> "ContentBasedRecommender":
        self.movie_ids = movies["item_id"].astype(str).tolist()
        self.movie_genres = dict(
            zip(movies["item_id"].astype(str), movies["genre"].astype(str), strict=True)
        )
        joined = train.merge(movies[["item_id", "genre"]], on="item_id", how="left")
        joined["preference_weight"] = (joined["rating"] - 3.0).clip(lower=0.1)
        totals = joined.groupby(["user_id", "genre"])["preference_weight"].sum()
        self.user_genre_scores = {}
        for (user_id, genre), score in totals.items():
            self.user_genre_scores.setdefault(str(user_id), {})[str(genre)] = float(score)
        self.popularity = PopularityRecommender().fit(train, self.movie_ids)
        return self

    def recommend(self, user_id: str, seen: set[str], k: int) -> list[str]:
        if self.popularity is None:
            raise RuntimeError("The content-based model must be fitted first")
        genre_scores = self.user_genre_scores.get(str(user_id))
        if not genre_scores:
            return self.popularity.recommend(user_id, seen, k)
        catalog = pd.DataFrame(
            {"item_id": self.movie_ids, "genre": [self.movie_genres[item] for item in self.movie_ids]}
        )
        catalog["preference"] = catalog["genre"].map(genre_scores).fillna(0.0)
        catalog["popularity_rank"] = catalog["item_id"].map(
            {movie: rank for rank, movie in enumerate(self.popularity.ranking)}
        )
        catalog = catalog[~catalog["item_id"].isin(seen)]
        catalog = catalog.sort_values(
            ["preference", "popularity_rank"], ascending=[False, True], kind="stable"
        )
        return catalog["item_id"].head(k).tolist()


def build_recommendations(model, test: pd.DataFrame, train: pd.DataFrame, k: int) -> dict[str, list[str]]:
    seen_by_user = train.groupby("user_id")["item_id"].apply(lambda values: set(values.astype(str)))
    return {
        str(user_id): model.recommend(
            str(user_id), seen_by_user.get(user_id, set()), k
        )
        for user_id in test["user_id"].astype(str).drop_duplicates()
    }
