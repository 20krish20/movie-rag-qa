import pandas as pd
from movie_rag.config.settings import MOVIES_CSV

def load_movies():
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {MOVIES_CSV}")
    return pd.read_csv(MOVIES_CSV)
