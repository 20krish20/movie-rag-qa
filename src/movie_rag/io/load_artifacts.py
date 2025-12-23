import pickle
import numpy as np
import faiss
from movie_rag.config import settings

def load_embeddings():
    if not settings.EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(f"Missing embeddings: {settings.EMBEDDINGS_NPY}")
    return np.load(settings.EMBEDDINGS_NPY)

def load_faiss_index():
    if not settings.FAISS_INDEX.exists():
        raise FileNotFoundError(f"Missing FAISS index: {settings.FAISS_INDEX}")
    return faiss.read_index(str(settings.FAISS_INDEX))

def load_movie_ids():
    if not settings.MOVIE_IDS_PKL.exists():
        raise FileNotFoundError(f"Missing movie_ids: {settings.MOVIE_IDS_PKL}")
    with open(settings.MOVIE_IDS_PKL, "rb") as f:
        return pickle.load(f)
