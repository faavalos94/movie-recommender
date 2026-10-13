"""Ranking metrics and the evaluation loop, shared by every notebook."""
import numpy as np
import pandas as pd


def recall_at_k(recommended, relevant, k):
    """Of everything the user liked, what fraction did we surface in the top k?"""
    if not relevant:
        return np.nan
    return len(set(recommended[:k]) & relevant) / len(relevant)


def precision_at_k(recommended, relevant, k):
    """Of the k items we recommended, what fraction did the user like?"""
    return len(set(recommended[:k]) & relevant) / k


def ndcg_at_k(recommended, relevant, k):
    """Like recall, but rewards putting hits near the top of the list."""
    if not relevant:
        return np.nan
    dcg = sum((1.0 if item in relevant else 0.0) / np.log2(i + 2)
              for i, item in enumerate(recommended[:k]))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else np.nan


def build_test_liked(test, like_threshold=4.0):
    """For each test user, the set of movies they rated highly."""
    liked = (test[test.rating >= like_threshold]
             .groupby("userId")["movieId"].apply(set))
    return liked[liked.apply(len) >= 1]


def score_users(recommend_fn, test_liked, k=10, n_users=None, seed=42):
    """Recall, precision and NDCG at k for each user, one row per user."""
    users = test_liked.index
    if n_users is not None and n_users < len(users):
        users = pd.Series(users).sample(n_users, random_state=seed).values

    rows = []
    for u in users:
        relevant = test_liked.loc[u]
        recs = recommend_fn(u, k)
        rows.append({
            "userId":    u,
            "recall":    recall_at_k(recs, relevant, k),
            "precision": precision_at_k(recs, relevant, k),
            "ndcg":      ndcg_at_k(recs, relevant, k),
        })
    return pd.DataFrame(rows).set_index("userId")


def evaluate(recommend_fn, test_liked, k=10, n_users=None, seed=42):
    """Average recall, precision and NDCG at k across users.
    
    recommend_fn(user_id, k) -> list of movieIds, best first
    """
    return score_users(recommend_fn, test_liked, k, n_users, seed).mean().round(4)
