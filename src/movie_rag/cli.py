import argparse

from movie_rag.config.settings import DEFAULT_TOP_K
from movie_rag.io.load_data import load_movies
from movie_rag.io.load_artifacts import load_faiss_index, load_movie_ids
from movie_rag.indexing.embedder import Embedder
from movie_rag.pipelines.router import answer_any_query
from movie_rag.llm import call_llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    rich_movies = load_movies()
    index = load_faiss_index()
    movie_ids = load_movie_ids()
    embedder = Embedder(device=args.device)

    out = answer_any_query(
        args.query,
        k=args.k,
        rich_movies=rich_movies,
        embedder=embedder,
        index=index,
        movie_ids=movie_ids,
        call_llm_fn=call_llm
    )

    print("\n" + "=" * 80)
    print("QUERY:", out["query"])
    print("TYPE:", out["query_type"], "| PIPELINE:", out["pipeline_used"])

    if out["query_type"] == "factual":
        print("\nGENERATED CODE:\n", out["generated_code"])
        print("\nRESULT:\n", out["result"])
    else:
        print("\nANSWER:\n", out["answer"])
        print("\nTOP TITLES:", [r["Title"] for r in out["retrieved"]])

if __name__ == "__main__":
    main()
