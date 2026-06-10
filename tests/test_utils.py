"""
Tests for app/utils.py — label parsing helpers.
"""
import pytest
from app.utils import parse_sentiment_explanation, parse_sentiment_label


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Sentiment: positive. This statement reflects favorable financial conditions.",
            "positive",
        ),
        (
            "Sentiment: negative. This statement reflects unfavorable financial conditions.",
            "negative",
        ),
        (
            "Sentiment: neutral. This statement reflects neutral financial conditions.",
            "neutral",
        ),
        # Case-insensitive matching
        ("Sentiment: Positive. Some text.", "positive"),
        ("SENTIMENT: NEGATIVE.", "negative"),
        # Label anywhere in text
        ("The answer is: Sentiment: neutral here.", "neutral"),
        # No recognizable label
        ("I have no idea what the sentiment is.", "unknown"),
        ("", "unknown"),
    ],
)
def test_parse_sentiment_label(text, expected):
    assert parse_sentiment_label(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Sentiment: positive. This statement reflects favorable financial conditions.",
            "This statement reflects favorable financial conditions.",
        ),
        (
            "Sentiment: negative. This statement reflects unfavorable financial conditions.",
            "This statement reflects unfavorable financial conditions.",
        ),
        (
            "Sentiment: neutral. This statement reflects neutral financial conditions.",
            "This statement reflects neutral financial conditions.",
        ),
        # Case-insensitive label
        ("Sentiment: Positive. Some explanation here.", "Some explanation here."),
        # No structured format — return empty string
        ("I have no idea what the sentiment is.", ""),
        ("", ""),
        # Label present but no trailing explanation
        ("Sentiment: negative.", ""),
    ],
)
def test_parse_sentiment_explanation(text, expected):
    assert parse_sentiment_explanation(text) == expected
