"""Precision@K, Recall@K, NDCG@K, catalog coverage, and genre diversity."""

from __future__ import annotations

import math

import pandas as pd


def evaluate(
    recommendations: dict[str, list[str]],
    test: pd.DataFrame,
    genres: dict[str, str],
    catalog_size: int,
    k: int,
    relevance_threshold: float,
) -> dict[str, float]:
    test = test.assign(user_id=test["user_id"].astype(str), item_id=test["item_id"].astype(str))
    relevant = (
        test[test["rating"] >= relevance_threshold]
        .groupby("user_id")["item_id"].apply(set).to_dict()
    )
    users = test["user_id"].drop_duplicates().tolist()
    if not relevant:
        raise ValueError("No relevant test ratings; lower the relevance threshold")
    precision_values, recall_values, ndcg_values, diversity_values = [], [], [], []
    recommended_catalog: set[str] = set()
    for user in users:
        ranked = recommendations.get(user, [])[:k]
        relevant_items = relevant.get(user, set())
        hits = [int(item in relevant_items) for item in ranked]
        precision_values.append(sum(hits) / k)
        recall_values.append(sum(hits) / len(relevant_items) if relevant_items else 0.0)
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_count = min(len(relevant_items), k)
        idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_count))
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        recommended_catalog.update(ranked)
        pairs = [
            genres.get(left) != genres.get(right)
            for index, left in enumerate(ranked)
            for right in ranked[index + 1 :]
        ]
        diversity_values.append(sum(pairs) / len(pairs) if pairs else 0.0)
    return {
        f"precision_at_{k}": sum(precision_values) / len(users),
        f"recall_at_{k}": sum(recall_values) / len(users),
        f"ndcg_at_{k}": sum(ndcg_values) / len(users),
        "coverage": len(recommended_catalog) / catalog_size if catalog_size else 0.0,
        "diversity": sum(diversity_values) / len(users),
        "evaluated_users": float(len(users)),
    }
