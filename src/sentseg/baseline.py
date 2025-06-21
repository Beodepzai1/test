def split(text: str):
    import re
    return re.split(r"(?<=[.!?])\s+", text.strip())
