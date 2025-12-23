import re
from movie_rag.pipelines.semantic import semantic_pipeline
from movie_rag.pipelines.factual import factual_pipeline

def classify_query_type(query: str) -> str:
    q = query.lower().strip()

    factual_keywords = [
        "average", "mean", "count", "how many", "number of", "total", "sum",
        "highest", "lowest", "maximum", "minimum", "max", "min", "top", "best", "worst",
        "greater than", "less than", "higher than", "lower than", "before", "after", "between"
    ]
    factual_cols = ["rating", "votes", "gross", "metascore", "runtime", "year"]

    if any(k in q for k in factual_keywords) and any(c in q for c in factual_cols):
        return "factual"

    if re.search(r"\b(19|20)\d{2}\b", q) and any(w in q for w in ["after","before","between","since","from"]):
        return "factual"

    return "semantic"

def answer_any_query(query: str, *, k: int, rich_movies, embedder, index, movie_ids, call_llm_fn):
    qtype = classify_query_type(query)

    if qtype == "factual":
        out = factual_pipeline(query, rich_movies=rich_movies, call_llm_fn=call_llm_fn)
        return {
            "query": query,
            "query_type": "factual",
            "pipeline_used": "factual_pipeline",
            **out
        }

    out = semantic_pipeline(query, k=k, embedder=embedder, index=index, movie_ids=movie_ids, rich_movies=rich_movies)
    return {
        "query": query,
        "query_type": "semantic",
        "pipeline_used": "semantic_pipeline",
        **out
    }
