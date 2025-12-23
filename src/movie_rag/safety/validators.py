import re

def validate_code(code: str):
    code = code.strip()

    # enforce single line
    if "\n" in code:
        raise ValueError("Factual code must be a SINGLE LINE: result = <expression>")

    # must assign result
    if not re.match(r"^result\s*=\s*.+$", code):
        raise ValueError("Code must match: result = <expression>")

    # must reference rich_movies
    if "rich_movies" not in code:
        raise ValueError("Code must reference rich_movies")

    # ban common dangerous / non-deterministic constructs
    forbidden = ["if ", "for ", "while ", "def ", "class ", "import ", "lambda", "try:", "except", "open("]
    if any(tok in code for tok in forbidden):
        raise ValueError("Forbidden construct in generated code")

    # ban placeholder
    if "..." in code:
        raise ValueError("Ellipsis (...) not allowed")
