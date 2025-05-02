"""
Tests for the evaluation metrics module.

This module contains tests for the metrics used to evaluate LLM performance.
"""

from src.evaluation.metrics import (
    analyze_results_by_category,
    calculate_metrics,
    extract_predicted_word,
    is_correct_answer,
    normalize_word,
)


def test_normalize_word():
    """Test word normalization function."""
    # Test lowercase conversion
    assert normalize_word("Apple") == "apple"

    # Test punctuation removal
    assert normalize_word("apple!") == "apple"
    assert normalize_word("apple.") == "apple"
    assert normalize_word("apple,") == "apple"

    # Test whitespace normalization
    assert normalize_word("  apple  ") == "apple"
    assert normalize_word("apple juice") == "apple juice"
    assert normalize_word("apple  juice") == "apple juice"


def test_is_correct_answer():
    """Test function for checking if an answer is correct."""
    # Test exact match
    assert is_correct_answer("apple", "apple")["exact"] is True
    assert is_correct_answer("Apple", "apple")["exact"] is True
    assert is_correct_answer("apple", "banana")["exact"] is False

    # Test with synonyms
    assert (
        is_correct_answer("fruit", "apple", synonyms=["fruit", "produce"])["synonym"]
        is True
    )
    assert (
        is_correct_answer("produce", "apple", synonyms=["fruit", "produce"])["synonym"]
        is True
    )
    assert (
        is_correct_answer("vegetable", "apple", synonyms=["fruit", "produce"])[
            "synonym"
        ]
        is False
    )

    # Test fuzzy matching
    assert is_correct_answer("app", "apple")["fuzzy"] is True
    assert is_correct_answer("applesauce", "apple")["fuzzy"] is True
    assert is_correct_answer("banana", "apple")["fuzzy"] is False

    # Test with list of predictions (top-k)
    assert is_correct_answer(["apple", "banana"], "apple")["exact"] is True
    assert is_correct_answer(["banana", "pear"], "apple")["exact"] is False
    assert (
        is_correct_answer(["fruit", "produce"], "apple", synonyms=["fruit", "produce"])[
            "synonym"
        ]
        is True
    )
    assert (
        is_correct_answer(
            ["vegetable", "grain"], "apple", synonyms=["fruit", "produce"]
        )["synonym"]
        is False
    )
    assert is_correct_answer(["app", "applesauce"], "apple")["fuzzy"] is True
    assert is_correct_answer(["banana", "pear"], "apple")["fuzzy"] is False


def test_extract_predicted_word():
    """Test function for extracting predicted words from LLM responses."""
    # Test extracting word in quotes
    assert extract_predicted_word('The word is "apple".') == "apple"

    # Test extracting word after common phrases
    assert extract_predicted_word("The word is apple.") == "apple"
    assert extract_predicted_word("I think the word is banana.") == "banana"
    assert extract_predicted_word("The answer would be cat.") == "cat"

    # Test fallback to first word
    assert extract_predicted_word("dog is a domesticated animal") == "dog"

    # Test empty response
    assert extract_predicted_word("") == ""


def test_calculate_metrics():
    """Test function for calculating evaluation metrics."""
    # Create test results (predictions as lists for top-k)
    results = [
        {
            "word": "apple",
            "definition": "A fruit",
            "prediction": ["apple"],
            "synonyms": ["fruit"],
        },
        {
            "word": "banana",
            "definition": "A yellow fruit",
            "prediction": ["banana"],
            "synonyms": ["fruit"],
        },
        {
            "word": "cat",
            "definition": "A feline animal",
            "prediction": ["dog"],
            "synonyms": ["feline"],
        },
        {
            "word": "dog",
            "definition": "A canine animal",
            "prediction": ["canine"],
            "synonyms": ["canine"],
        },
    ]

    metrics = calculate_metrics(results)
    assert metrics["exact_accuracy"] == 0.5  # 2 out of 4 correct
    assert metrics["synonym_accuracy"] == 0.75
    assert metrics["fuzzy_accuracy"] == 0.5
    assert metrics["num_samples"] == 4

    results_topk = [
        {
            "word": "apple",
            "definition": "A fruit",
            "prediction": ["pear", "apple", "banana"],
            "synonyms": ["fruit"],
        },
        {
            "word": "banana",
            "definition": "A yellow fruit",
            "prediction": ["banana", "apple", "fruit"],
            "synonyms": ["fruit"],
        },
        {
            "word": "cat",
            "definition": "A feline animal",
            "prediction": ["dog", "wolf", "fox"],
            "synonyms": ["feline"],
        },
        {
            "word": "dog",
            "definition": "A canine animal",
            "prediction": ["canine", "wolf", "fox"],
            "synonyms": ["canine"],
        },
    ]
    metrics_topk = calculate_metrics(results_topk)
    assert (
        metrics_topk["exact_accuracy"] == 0.25
    )  # 3 out of 4 have the correct word in top-k
    assert (
        metrics_topk["synonym_accuracy"] == 0.50
    )  # 4 out of 4 have correct or synonym in top-k


def test_analyze_results_by_category():
    """Test function for analyzing results by category."""
    # Create test results with categories
    results = [
        {
            "word": "apple",
            "definition": "A fruit",
            "prediction": ["apple"],
            "part_of_speech": "noun",
            "synonyms": [],
        },
        {
            "word": "banana",
            "definition": "A yellow fruit",
            "prediction": ["banana"],
            "part_of_speech": "noun",
            "synonyms": [],
        },
        {
            "word": "run",
            "definition": "To move quickly",
            "prediction": ["sprint"],
            "part_of_speech": "verb",
            "synonyms": ["sprint"],
        },
        {
            "word": "happy",
            "definition": "Feeling joy",
            "prediction": ["sad"],
            "part_of_speech": "adjective",
            "synonyms": [],
        },
    ]

    # Test analysis by part of speech
    category_metrics = analyze_results_by_category(results)

    # Check that categories were created correctly
    assert "by_part_of_speech" in category_metrics
    assert "by_word_length" in category_metrics

    # Check metrics by part of speech
    pos_metrics = category_metrics["by_part_of_speech"]
    assert "noun" in pos_metrics
    assert "verb" in pos_metrics
    assert "adjective" in pos_metrics

    # Check accuracy by part of speech
    assert pos_metrics["noun"]["exact_accuracy"] == 1.0  # 2 out of 2 correct
    assert pos_metrics["verb"]["synonym_accuracy"] == 1.0  # 1 out of 1 with synonyms
    assert pos_metrics["adjective"]["exact_accuracy"] == 0.0  # 0 out of 1 correct
