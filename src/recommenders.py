"""The two recommenders from notebook 02, packaged so other notebooks and the app can use them."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def build_recommenders(train, like_threshold=4.0):
    """Build both models from the training ratings.
    
    Returns a dict of functions, each with the signature (user_id, k) -> list of movieIds,
    plus R_liked and u_index for analysis.
    """
    # One row per user, one column per movie
    user_ids = np.sort(train.userId.unique())
    item_ids = np.sort(train.movieId.unique())
    u_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_index = pd.Series(np.arange(len(item_ids)), index=item_ids)

    def to_matrix(df):
        return csr_matrix(
            (np.ones(len(df), dtype="float32"),
             (u_index[df.userId].values, i_index[df.movieId].values)),
             shape=(len(user_ids), len(item_ids)))

    R_seen = to_matrix(train)
    R_liked = to_matrix(train[train.rating >= like_threshold])
    R_norm = normalize(R_liked, axis=0)
    pop_scores = np.asarray(R_liked.sum(axis=0)).ravel()

    def top_k_unseen(scores, user_id, k):
        scores = scores.astype("float64").copy()
        scores[R_seen[u_index[user_id]].indices] = -np.inf
        return item_ids[np.argsort(-scores)[:k]].tolist()

    def popularity_rec(user_id, k):
        return top_k_unseen(pop_scores, user_id, k)

    def itemitem_rec(user_id, k):
        liked = np.asarray(R_liked[u_index[user_id]].todense()).ravel()
        scores = R_norm.T @ (R_norm @ liked)
        return top_k_unseen(scores, user_id, k)

    return {
        "Popularity": popularity_rec,
        "Item-item": itemitem_rec,
        "R_liked": R_liked,
        "u_index": u_index,
    }
