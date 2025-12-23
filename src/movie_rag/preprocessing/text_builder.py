import pandas as pd

# NOTE: This is your original text-building logic (kept as-is, just wrapped in a function).
def build_movie_text(row):

    # Part 1
    part1 = []
    title = str(row["Title"]).strip()
    year = int(row["Year"]) if not pd.isna(row["Year"]) else None
    genre = str(row["Genres"]).strip()

    if title:
        if year:
            part1.append(f"{title} ({year}) is a ")
        else:
            part1.append(f"{title} is a ")
    if genre and genre.lower() != "unknown":
        part1.append(f"{genre} movie.")
    else:
        part1.append("movie.")

    part1_sentence = "".join(part1)

    # Part 2
    part2_sentence = ""
    director = str(row["Director"]).strip()
    stars = str(row["Stars"]).strip()

    if director and director.lower() != "unknown":
        part2_sentence += f" It is directed by {director}"
    if stars and stars.lower() != "unknown":
        part2_sentence += f" and stars {stars}."

    # Part 3
    part3 = []
    rating = row["Rating"]
    votes = row["Votes"]
    gross = row["Gross(Million)"]

    if not pd.isna(rating):
        part3.append(f"an IMDB rating of {rating} out of 10")
    if not pd.isna(votes):
        part3.append(f"total votes of {votes}")
    if not pd.isna(gross):
        part3.append(f"a gross total of approximately {gross} million dollars")

    part3_sentence = ""
    if part3:
        part3_sentence = " It has " + ", ".join(part3[:-1])
        if len(part3) > 1:
            part3_sentence += ", and " + part3[-1] + "."
        else:
            part3_sentence += "."

    # Part 4
    certificate = str(row["Certificate"]).strip()
    runtime = row["Runtime"]
    metascore = row["Metascore"]

    part4 = []
    if certificate and certificate.lower() != "unknown":
        part4.append(f"rated {certificate}")
    if not pd.isna(runtime):
        part4.append(f"a runtime of {int(runtime)} minutes")

    part4_sentence = ""
    if part4:
        part4_sentence = " It is " + " with ".join(part4) + "."

    # Part 5
    summary = str(row["Summary"]).strip()
    part5_sentence = ""
    if summary:
        part5_sentence = f" Plot summary: {summary}"

    full_text = part1_sentence + part2_sentence + part3_sentence + part4_sentence + part5_sentence
    return full_text.strip()
