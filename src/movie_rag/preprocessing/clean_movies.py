import pandas as pd
import numpy as np

def clean_movies(movie_data: pd.DataFrame) -> pd.DataFrame:
    movies = movie_data.copy()

    # Fill missing as 'Unknown' in Stars and Certificate Columns (your original logic)
    movies["Stars"] = movies["Stars"].fillna("Unknown")
    movies["Certificate"] = movies["Certificate"].fillna("Unknown")

    # Ensure numeric types (robust)
    for col in ["Year","Runtime","Rating","Metascore","Votes","Gross(Million)"]:
        if col in movies.columns:
            movies[col] = pd.to_numeric(movies[col], errors="coerce")

    # Fill missing in Runtime column with its mean (your original logic)
    movies["Runtime"] = movies["Runtime"].fillna(movies["Runtime"].mean())

    # Fill missing in Gross(Million) column with its nonzero median (your original logic)
    gross_nonzero = movies["Gross(Million)"][movies["Gross(Million)"] > 0]
    median_gross = gross_nonzero.median()
    movies["Gross(Million)"] = movies["Gross(Million)"].fillna(median_gross)

    # Basic whitespace cleanup (safe)
    for col in ["Title","Genres","Director","Stars","Certificate","Summary"]:
        if col in movies.columns:
            movies[col] = movies[col].astype("string").fillna("").str.strip().replace({"": np.nan})

    # Restore Unknown for key categoricals if stripping made them NaN
    movies["Stars"] = movies["Stars"].fillna("Unknown")
    movies["Certificate"] = movies["Certificate"].fillna("Unknown")

    return movies
