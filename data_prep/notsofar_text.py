"""Text normalization shared by the NotSoFar preparation and dictionary tools."""

import re

_ANNOTATION_TAG = re.compile(r"<\s*(?:ST|FILL|UNKNOWN)\s*/\s*>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """Remove NotSoFar annotation tags and return canonical ASR text.

    The returned text is intentionally word based.  The existing ASR datasets
    uppercase text and insert the ``|`` character during ``process_trans``;
    storing ``|`` here would therefore encode the word boundary twice.
    """
    text = _ANNOTATION_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text.upper()


def transcript_tokens(text: str) -> list[str]:
    """Return Character-CTC tokens for a normalized NotSoFar transcript."""
    normalized = normalize_transcript(text)
    if not normalized:
        return ["|"]
    tokens = list(normalized.replace(" ", "|"))
    if tokens[-1] != "|":
        tokens.append("|")
    return tokens
