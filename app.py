"""Movie recommender demo.

Pick movies you like and compare two recommenders side by side:
a popularity list that ignores your picks, and item-item collaborative filtering.
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

REPO = "https://github.com/faavalos94/movie-recommender"
K = 10
DEFAULTS = ["Toy Story (1995)", "Finding Nemo (2003)", "Shrek (2001)"]


@st.cache_data
def load():
    movies = pd.read_csv("demo/movies.csv")
    neighbors = pd.read_csv("demo/neighbors.csv")
    return movies, neighbors


movies, neighbors = load()
title_of = dict(zip(movies.movieId, movies.title))
genres_of = dict(zip(movies.movieId, movies.genres.str.replace("|", ", ", regex=False)))
id_of = dict(zip(movies.title, movies.movieId))


def popularity_recs(picked_ids, k=K):
    """The most-liked movies, skipping the ones you picked."""
    ranked = movies.sort_values("likes", ascending=False)
    return ranked[~ranked.movieId.isin(picked_ids)].movieId.head(k).tolist()


def itemitem_recs(picked_ids, k=K):
    """Score each movie by its total similarity to your picks.

    Returns (movieId, the pick that contributed most) pairs.
    """
    near = neighbors[neighbors.movieId.isin(picked_ids) & ~neighbors.neighborId.isin(picked_ids)]
    scores = near.groupby("neighborId").similarity.sum().sort_values(ascending=False)
    because = (near.sort_values("similarity", ascending=False)
                   .drop_duplicates("neighborId")
                   .set_index("neighborId").movieId)
    recs = [(m, because[m]) for m in scores.index[:k]]

    # Rarely, a niche pick has too few neighbors; top up from popularity
    for m in popularity_recs(picked_ids + [r for r, _ in recs], k - len(recs)):
        recs.append((m, None))
    return recs


# ------------------------------------------------------------------- header
st.title("Movie recommender")
st.caption("Pick a few movies you like. Compare a one-size-fits-all list "
           "with recommendations built from your picks.")

picked_titles = st.multiselect(
    "Movies you like",
    options=movies.sort_values("likes", ascending=False).title.tolist(),
    default=[t for t in DEFAULTS if t in id_of],
    placeholder="Start typing a title…",
)
picked = [int(id_of[t]) for t in picked_titles]

if not picked:
    st.info("Pick at least one movie to see recommendations.")
    st.stop()

pop = popularity_recs(picked)
ii = itemitem_recs(picked)

# -------------------------------------------------------------------- lists
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Most popular")
    st.caption("The same list for everyone. Ignores your picks.")
    for i, m in enumerate(pop, 1):
        st.markdown(f"**{i}. {title_of[m]}**  \n"
                    f"<span style='color:gray;font-size:0.85em'>{genres_of[m]}</span>",
                    unsafe_allow_html=True)

with right:
    st.subheader("Recommended for you")
    st.caption("Item-item: movies liked by the same people who liked your picks.")
    for i, (m, src) in enumerate(ii, 1):
        reason = f"because you liked {title_of[src]}" if src is not None else "popular pick"
        st.markdown(f"**{i}. {title_of[m]}**  \n"
                    f"<span style='color:gray;font-size:0.85em'>{reason}</span>",
                    unsafe_allow_html=True)

overlap = len(set(pop) & {m for m, _ in ii})
st.caption(f"{overlap} of {K} movies appear in both lists.")

st.divider()

# ---------------------------------------------------------------- findings
st.subheader("What the evaluation found")
st.markdown("""
Both models were trained on ratings before 2018 and tested on what users went on to like
in 2018–2019. A rating of 4+ counts as "liked."

- **Personalization helps, but less than you'd think.** Item-item scored about 45% higher
  NDCG@10 than popularity (0.11 vs 0.08). Yet popularity got two-thirds of the way there
  while knowing nothing about the user.
- **Item-item helps mid-history users most.** The gain peaked at +61% for users with
  51–150 liked movies.
- **Heavy users are harder to recommend for.** Only 5% of their next likes were among the
  100 most popular movies, against 24% for light users. Their tastes are more niche.
""")
st.markdown(f"[Full write-up and code on GitHub]({REPO})")
st.caption("Data: MovieLens 25M (GroupLens). The demo uses the 3,000 most-liked movies.")
