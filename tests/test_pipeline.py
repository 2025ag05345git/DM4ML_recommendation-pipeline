from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.metrics import evaluate
from src.ingestion.generate_synthetic import generate
from src.preparation.prepare import prepare
from src.training.recommenders import ContentBasedRecommender, PopularityRecommender
from src.validation.validate import validate_frames


class SyntheticCatalogTests(unittest.TestCase):
    def test_default_catalog_has_5000_english_movies(self) -> None:
        users, movies, interactions = generate()

        self.assertEqual(len(movies), 5000)
        self.assertEqual(movies["item_id"].nunique(), 5000)
        self.assertTrue(movies["language"].eq("English").all())
        self.assertTrue(movies["language_code"].eq("en").all())
        self.assertTrue(movies["title"].str.contains(r"[A-Za-z]").all())
        self.assertEqual(len(users), 500)
        self.assertEqual(len(interactions), 16005)
        invalid, report, valid, valid_users, valid_movies = validate_frames(
            users, movies, interactions
        )
        self.assertEqual(len(valid_movies), 5000)
        self.assertEqual(len(valid_users), 500)
        self.assertEqual(len(valid), 16000)
        self.assertEqual(len(invalid), 5)
        self.assertEqual(report["row_counts"]["movies_valid"], 5000)


class ValidationTests(unittest.TestCase):
    def test_invalid_ratings_duplicates_and_foreign_keys_are_quarantined(self) -> None:
        users = pd.DataFrame(
            {"user_id": ["u1"], "age": [30], "occupation": ["teacher"], "preferred_genre": ["Drama"]}
        )
        movies = pd.DataFrame(
            {
                "item_id": ["m1"], "title": ["Film"], "genre": ["Drama"],
                "release_year": [2020], "language": ["English"], "language_code": ["en"],
            }
        )
        interactions = pd.DataFrame(
            [
                {"interaction_id": "i1", "user_id": "u1", "item_id": "m1", "rating": 5, "timestamp": "2024-01-01"},
                {"interaction_id": "i1", "user_id": "u1", "item_id": "m1", "rating": 5, "timestamp": "2024-01-01"},
                {"interaction_id": "i2", "user_id": "u1", "item_id": "m1", "rating": 7, "timestamp": "2024-01-02"},
                {"interaction_id": "i3", "user_id": "absent", "item_id": "m1", "rating": 4, "timestamp": "2024-01-03"},
            ]
        )

        invalid, report, valid, _, _ = validate_frames(users, movies, interactions)

        self.assertEqual(len(invalid), 3)
        self.assertEqual(len(valid), 1)
        self.assertEqual(report["quarantine_reasons"]["duplicate_interaction_id"], 1)
        self.assertEqual(report["quarantine_reasons"]["rating_out_of_range"], 1)
        self.assertEqual(report["quarantine_reasons"]["unknown_user"], 1)

    def test_non_english_movie_rows_are_quarantined(self) -> None:
        users = pd.DataFrame(
            {"user_id": ["u1"], "age": [30], "occupation": ["teacher"], "preferred_genre": ["Drama"]}
        )
        movies = pd.DataFrame(
            {
                "item_id": ["m1", "m2"], "title": ["Film", "Película"],
                "genre": ["Drama", "Drama"], "release_year": [2020, 2021],
                "language": ["English", "Spanish"], "language_code": ["en", "es"],
            }
        )
        interactions = pd.DataFrame(
            [{"interaction_id": "i1", "user_id": "u1", "item_id": "m2",
              "rating": 5, "timestamp": "2024-01-01"}]
        )

        invalid, report, valid, _, valid_movies = validate_frames(
            users, movies, interactions
        )

        self.assertEqual(len(valid_movies), 1)
        self.assertEqual(valid_movies.iloc[0]["item_id"], "m1")
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(report["quarantine_reasons"]["non_english_language"], 1)
        self.assertEqual(report["quarantine_reasons"]["invalid_language_code"], 1)


class PreparationTests(unittest.TestCase):
    def test_temporal_holdout_has_no_cutoff_timestamp_leakage(self) -> None:
        users = pd.DataFrame(
            {"user_id": ["u1"], "age": [30], "occupation": ["teacher"], "preferred_genre": ["Drama"]}
        )
        movies = pd.DataFrame(
            {
                "item_id": ["m1", "m2", "m3"], "title": ["A", "B", "C"],
                "genre": ["Drama", "Comedy", "Action"], "release_year": [2020, 2021, 2022],
                "language": ["English"] * 3, "language_code": ["en"] * 3,
            }
        )
        interactions = pd.DataFrame(
            [
                {"interaction_id": "i1", "user_id": "u1", "item_id": "m1", "rating": 4, "timestamp": "2024-01-01"},
                {"interaction_id": "i2", "user_id": "u1", "item_id": "m2", "rating": 4, "timestamp": "2024-02-01"},
                {"interaction_id": "i3", "user_id": "u1", "item_id": "m3", "rating": 5, "timestamp": "2024-02-01"},
            ]
        )

        _, _, train, test = prepare(users, movies, interactions, 1)

        self.assertEqual(train["interaction_id"].tolist(), ["i1"])
        self.assertEqual(set(test["interaction_id"]), {"i2", "i3"})
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())


class RecommenderAndMetricsTests(unittest.TestCase):
    def test_models_recommend_unseen_movies_and_metrics_are_bounded(self) -> None:
        movies = pd.DataFrame(
            {
                "item_id": ["m1", "m2", "m3", "m4"],
                "genre": ["Drama", "Drama", "Comedy", "Action"],
            }
        )
        train = pd.DataFrame(
            [
                {"interaction_id": "i1", "user_id": "u1", "item_id": "m1", "rating": 5},
                {"interaction_id": "i2", "user_id": "u1", "item_id": "m2", "rating": 4},
                {"interaction_id": "i3", "user_id": "u2", "item_id": "m1", "rating": 5},
                {"interaction_id": "i4", "user_id": "u2", "item_id": "m3", "rating": 3},
            ]
        )
        test = pd.DataFrame(
            [{"user_id": "u1", "item_id": "m3", "rating": 5},
             {"user_id": "u2", "item_id": "m4", "rating": 2}]
        )
        models = {
            "popularity": PopularityRecommender().fit(train, movies.item_id.tolist()),
            "content": ContentBasedRecommender().fit(train, movies),
        }
        for model in models.values():
            recs = model.recommend("u1", {"m1", "m2"}, 2)
            self.assertEqual(len(recs), 2)
            self.assertFalse({"m1", "m2"}.intersection(recs))
            metrics = evaluate(
                {"u1": recs, "u2": model.recommend("u2", {"m1", "m3"}, 2)},
                test, dict(zip(movies.item_id, movies.genre, strict=True)), 4, 2, 4,
            )
            for name in ("precision_at_2", "recall_at_2", "ndcg_at_2", "coverage", "diversity"):
                self.assertGreaterEqual(metrics[name], 0.0)
                self.assertLessEqual(metrics[name], 1.0)


if __name__ == "__main__":
    unittest.main()
