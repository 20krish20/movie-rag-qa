import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from movie_rag.config import settings

class Embedder:
    def __init__(self, model_name: str = settings.EMBED_MODEL_NAME, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], convert_to_numpy=True)
        emb = emb / np.clip(norm(emb, axis=1, keepdims=True), 1e-9, None)
        return emb.astype("float32")

    def encode_corpus(self, texts, batch_size: int = 128, show_progress_bar: bool = True) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        ).astype("float32")
        emb = emb / np.clip(norm(emb, axis=1, keepdims=True), 1e-9, None)
        return emb
