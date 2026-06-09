import re

_LABEL_PATTERN = re.compile(r"Sentiment:\s*(positive|neutral|negative)", re.IGNORECASE)


def parse_sentiment_label(text: str) -> str:
    """Extract the sentiment label from a model-generated response.

    Returns 'positive', 'neutral', or 'negative' on a match,
    or 'unknown' if the text does not contain a recognizable label.
    """
    match = _LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return "unknown"
