from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MOVIES_CSV = DATA_DIR / "rich_movies.csv"

EMBEDDINGS_NPY = ARTIFACTS_DIR / "embeddings_norm.npy"
FAISS_INDEX = ARTIFACTS_DIR / "faiss.index"
MOVIE_IDS_PKL = ARTIFACTS_DIR / "movie_ids.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TOP_K = 5
