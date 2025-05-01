"""
Metrics module for evaluating LLM performance on word definition tasks.

This module provides functions for calculating various metrics to evaluate
how well LLMs understand word definitions, including handling synonyms.
"""

import json
import re

import pandas as pd


def normalize_word(word: str) -> str:
    """
    Normalize a word for comparison.

    Args:
        word: Word to normalize

    Returns:
        Normalized word
    """
    # Convert to lowercase
    word = word.lower()

    # Remove punctuation and special characters
    word = re.sub(r"[^\w\s]", "", word)

    # Remove extra whitespace
    word = re.sub(r"\s+", " ", word).strip()

    return word


def is_correct_answer(
    prediction: str,
    target: str,
    synonyms: list[str] | None = None,
    fuzzy_match: bool = False,
) -> bool:
    """
    Check if the prediction is correct, considering synonyms.

    Args:
        prediction: The predicted word
        target: The target word
        synonyms: List of acceptable synonyms
        fuzzy_match: Whether to allow fuzzy matching

    Returns:
        True if the prediction is correct, False otherwise
    """
    # Normalize words for comparison
    prediction = normalize_word(prediction)
    target = normalize_word(target)

    # Check exact match
    if prediction == target:
        return True

    # Check synonyms if provided
    if synonyms:
        normalized_synonyms = [normalize_word(syn) for syn in synonyms]
        if prediction in normalized_synonyms:
            return True

    # Check fuzzy match if enabled
    if fuzzy_match:
        # Simple fuzzy matching: check if one is a substring of the other
        if (prediction in target) or (target in prediction):
            return True

        # Could add more sophisticated fuzzy matching here (e.g., Levenshtein distance)

    return False


def extract_predicted_word(response: str) -> str:
    """
    Extract the predicted word from an LLM response.

    Args:
        response: The LLM's response text

    Returns:
        The extracted word
    """

    # Try to find a word in quotes
    quote_match = re.search(r'"([^"]+)"', response)
    if quote_match:
        return quote_match.group(1)

    # Try to find a word after common phrases
    phrase_matches = [
        re.search(
            r'(?:word is|answer is|word would be|answer would be)\s+["\']?([a-zA-Z]+)["\']?',
            response,
            re.IGNORECASE,
        ),
        re.search(
            r'(?:I think the word is|I believe the word is)\s+["\']?([a-zA-Z]+)["\']?',
            response,
            re.IGNORECASE,
        ),
    ]

    for match in phrase_matches:
        if match:
            return match.group(1)

    # If no clear pattern, return the first word (fallback)
    words = re.findall(r"\b[a-zA-Z]+\b", response)
    if words:
        return words[0]

    return ""


def calculate_metrics(
    results: list[dict[str, str | list[str]]],
    include_synonyms: bool = True,
    fuzzy_match: bool = False,
) -> dict[str, float]:
    """
    Calculate evaluation metrics for LLM performance.

    Args:
        results: List of dictionaries with keys 'word', 'definition', 'prediction', and optionally 'synonyms'
        include_synonyms: Whether to consider synonyms as correct answers
        fuzzy_match: Whether to allow fuzzy matching

    Returns:
        Dictionary of metrics
    """
    y_true = []
    y_pred = []

    for result in results:
        target = result["word"]
        prediction = result["prediction"]

        y_true.append(target)
        y_pred.append(prediction)

    # Calculate exact match accuracy (without synonyms)
    exact_matches = [
        normalize_word(pred) == normalize_word(true)
        for pred, true in zip(y_pred, y_true, strict=False)
    ]
    exact_accuracy = sum(exact_matches) / len(exact_matches)

    # Calculate accuracy with synonyms if requested
    if include_synonyms or fuzzy_match:
        synonym_matches = [
            is_correct_answer(
                prediction=results[i]["prediction"],
                target=results[i]["word"],
                synonyms=results[i].get("synonyms", []) if include_synonyms else None,
                fuzzy_match=fuzzy_match,
            )
            for i in range(len(results))
        ]
        synonym_accuracy = sum(synonym_matches) / len(synonym_matches)
    else:
        synonym_accuracy = exact_accuracy

    return {
        "exact_accuracy": exact_accuracy,
        "synonym_accuracy": synonym_accuracy,
        "num_samples": len(results),
    }


def analyze_results_by_category(
    results: list[dict[str, str | list[str]]], include_synonyms: bool = True
) -> dict[str, dict[str, float]]:
    """
    Analyze results by word category (e.g., part of speech, word length).

    Args:
        results: List of dictionaries with evaluation results
        include_synonyms: Whether to consider synonyms as correct answers

    Returns:
        Dictionary of metrics by category
    """
    df = pd.DataFrame(results)

    # Add word length column
    df["word_length"] = df["word"].str.len()

    # Define word length categories
    df["length_category"] = pd.cut(
        df["word_length"],
        bins=[0, 4, 7, 10, 100],
        labels=["very_short", "short", "medium", "long"],
    )

    # Calculate metrics by part of speech
    pos_metrics = {}
    if "part_of_speech" in df.columns:
        for pos in df["part_of_speech"].unique():
            if pd.isna(pos) or pos == "":
                continue

            pos_results = df[df["part_of_speech"] == pos].to_dict("records")
            if pos_results:
                pos_metrics[pos] = calculate_metrics(pos_results, include_synonyms)

    # Calculate metrics by word length category
    length_metrics = {}
    for category in df["length_category"].unique():
        if pd.isna(category):
            continue

        category_results = df[df["length_category"] == category].to_dict("records")
        if category_results:
            length_metrics[str(category)] = calculate_metrics(
                category_results, include_synonyms
            )

    return {"by_part_of_speech": pos_metrics, "by_word_length": length_metrics}


def save_evaluation_results(
    results: list[dict[str, str | list[str]]],
    metrics: dict[str, float],
    category_metrics: dict[str, dict[str, float]],
    output_file: str,
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results: List of dictionaries with evaluation results
        metrics: Dictionary of overall metrics
        category_metrics: Dictionary of metrics by category
        output_file: Path to save the results
    """
    output = {
        "overall_metrics": metrics,
        "category_metrics": category_metrics,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Evaluation results saved to {output_file}")
