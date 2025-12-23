import re
from movie_rag.safety.validators import validate_code

def build_factual_prompt(query: str) -> str:
    return f"""Return ONLY one Python code block (and nothing else).

Rules:
- Use ONLY the pandas DataFrame `rich_movies`
- Do NOT print
- Do NOT explain
- Output ONE LINE ONLY
- Assign the final answer to variable `result`
- Do NOT use ellipsis (...)

Columns in rich_movies:
Title, Year, Genres, Certificate, Runtime, Rating, Metascore,
Votes, Gross(Million), Director, Stars

Question:
{query}

Output format:
```python
result = <single pandas expression using rich_movies>
```
""".strip()

def extract_code_from_response(response_text: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise ValueError("No ```python``` block found from LLM.")
    return blocks[-1].strip()

def run_factual_code(code: str, rich_movies):
    local_env = {"rich_movies": rich_movies}
    exec(code, {}, local_env)
    return local_env.get("result", None)

def factual_pipeline(query: str, *, rich_movies, call_llm_fn):
    prompt = build_factual_prompt(query)

    raw = call_llm_fn(prompt)
    code = extract_code_from_response(raw)

    # retry if invalid or placeholder
    if "..." in code or "result = ..." in code:
        raw = call_llm_fn(prompt + "\nSTRICT: Do NOT use '...'. Return one real pandas line.")
        code = extract_code_from_response(raw)

    validate_code(code)
    result = run_factual_code(code, rich_movies)

    return {"query": query, "generated_code": code, "result": result}
