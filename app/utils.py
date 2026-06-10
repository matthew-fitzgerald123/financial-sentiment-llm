import re

_LABEL_PATTERN = re.compile(r"Sentiment:\s*(positive|neutral|negative)", re.IGNORECASE)
_EXPLANATION_PATTERN = re.compile(
    r"Sentiment:\s*(?:positive|neutral|negative)\.\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_sentiment_label(text: str) -> str:
    """Extract the sentiment label from a model-generated response.

    Returns 'positive', 'neutral', or 'negative' on a match,
    or 'unknown' if the text does not contain a recognizable label.
    """
    match = _LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return "unknown"


def parse_sentiment_explanation(text: str) -> str:
    """Extract the explanation text that follows the sentiment label.

    Returns the explanation string (e.g. 'This statement reflects favorable
    financial conditions.') or an empty string if not found.
    """
    match = _EXPLANATION_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return ""
