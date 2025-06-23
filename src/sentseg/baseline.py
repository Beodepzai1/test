def split(text: str):
    import re
    """Simple regex based sentence splitter."""
    return re.split(r"(?<=[.!?])\s+", text.strip())


def split_none(text: str):
    """Return the input text as a single sentence without segmentation."""
    text = text.strip()
    return [text] if text else []
