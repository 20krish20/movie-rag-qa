from typing import List, Dict
import numpy as np

def retrieve_movies_for_query(query: str, *, k: int, embedder, index, movie_ids, rich_movies) -> List[Dict]:
    q_emb = embedder.encode_query(query)
    distances, indices = index.search(q_emb, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        movie_row_idx = movie_ids[idx]
        row = rich_movies.loc[movie_row_idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "row_idx": int(movie_row_idx),
            "Title": row["Title"],
            "Year": int(row["Year"]) if not np.isnan(row["Year"]) else None,
            "Genres": row.get("Genres", ""),
            "Rating": float(row["Rating"]) if not np.isnan(row["Rating"]) else None,
            "Summary": row.get("Summary", "")
        })
    return results

def semantic_pipeline(query: str, *, k: int, embedder, index, movie_ids, rich_movies):
    retrieved = retrieve_movies_for_query(
        query, k=k, embedder=embedder, index=index, movie_ids=movie_ids, rich_movies=rich_movies
    )

    answer = "Top matches:\n" + "\n".join(
        [f"{r['rank']}. {r['Title']} ({r['Year']}) — score={r['score']:.3f}" for r in retrieved]
    )

    return {"query": query, "retrieved": retrieved, "answer": answer}
