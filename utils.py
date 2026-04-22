import re

def clean_text(text):
    """
    Cleans text to match the character set found in your Parakeet-TDT vocab.
    Preserves: a-z, A-Z, 0-9 (if in vocab), spaces
    """
    if not text:
        return ""

    # 1. Replace newlines/tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")

    # 2. Strip everything except letters (a-z, A-Z), spaces, and apostrophes
    # We remove . , ! ? and other symbols
    text = re.sub(r"[^a-zA-Z\s']", "", text)

    # 3. Collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text