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
    prediction: str | list[str],
    target: str,
    synonyms: list[str] | None = None,
) -> dict[str, bool]:
    """
    Check if the prediction is correct, considering synonyms.

    Args:
        prediction: The predicted word
        target: The target word
        synonyms: List of acceptable synonyms

    Returns:
        Dictionary with keys 'exact', 'synonym', and 'fuzzy' 
            indicating whether the prediction is correct for each metric
            Synonym and fuzzy are also correct if exact is correct
    """
    # Always treat prediction as a list for compatibility
    preds = prediction if isinstance(prediction, list) else [prediction]
    target = normalize_word(target)
    normalized_synonyms = [normalize_word(syn) for syn in synonyms] if synonyms else []
    is_correct = {"exact": False, "synonym": False, "fuzzy": False}
    for pred in preds:
        pred_norm = normalize_word(pred)
        # Check exact match
        if pred_norm == target:
            is_correct["exact"] = True
            # If exact match, synonym and fuzzy are also correct
            is_correct["synonym"] = True
            is_correct["fuzzy"] = True
        # Check synonyms if provided
        if not is_correct["synonym"] and normalized_synonyms and pred_norm in normalized_synonyms:
            is_correct["synonym"] = is_correct["exact"] or True
        # Check fuzzy match if enabled
        if not is_correct["fuzzy"]:
            if (pred_norm in target) or (target in pred_norm):
                is_correct["fuzzy"] = is_correct["exact"] or True
    return is_correct


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
) -> dict[str, float]:
    """
    Calculate evaluation metrics for LLM performance.

    Args:
        results: List of dictionaries with keys 'word', 'definition', 'prediction', and optionally 'synonyms'

    Returns:
        Dictionary of metrics
    """
    y_true = []
    y_pred = []

    for result in results:
        target = result["word"]
        prediction = result["prediction"][0]
        y_true.append(target)
        y_pred.append(prediction)

    # Calculate exact match accuracy (any of top-k predictions)
    is_correct = [
        is_correct_answer(
            prediction=results[i]["prediction"][0],
            target=results[i]["word"],
            synonyms=results[i].get("synonyms", []),
        )
        for i in range(len(results))
    ]
    
    exact_accuracy = sum(is_correct[i]["exact"] for i in range(len(is_correct))) / len(is_correct)
    synonym_accuracy = sum(is_correct[i]["synonym"] for i in range(len(is_correct))) / len(is_correct)
    fuzzy_accuracy = sum(is_correct[i]["fuzzy"] for i in range(len(is_correct))) / len(is_correct)

    return {
        "exact_accuracy": exact_accuracy,
        "synonym_accuracy": synonym_accuracy,
        "fuzzy_accuracy": fuzzy_accuracy,
        "num_samples": len(results),
    }


def analyze_results_by_category(
    results: list[dict[str, str | list[str]]],
) -> dict[str, dict[str, float]]:
    """
    Analyze results by word category (e.g., part of speech, word length).

    Args:
        results: List of dictionaries with evaluation results

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
                pos_metrics[pos] = calculate_metrics(pos_results)

    # Calculate metrics by word length category
    length_metrics = {}
    for category in df["length_category"].unique():
        if pd.isna(category):
            continue

        category_results = df[df["length_category"] == category].to_dict("records")
        if category_results:
            length_metrics[str(category)] = calculate_metrics(category_results)

    return {"by_part_of_speech": pos_metrics, "by_word_length": length_metrics}


def is_correct_topk(
    predictions: list[str],
    target: str,
) -> tuple[bool, bool]:
    """
    Check if the target is in the top-k predictions (exact and fuzzy).
    Returns (exact_found, fuzzy_found).
    """
    # Normalize target and synonyms
    target_norm = normalize_word(target)
    exact_found = False
    fuzzy_found = False
    for pred in predictions:
        pred_norm = normalize_word(pred)
        # Exact match or synonym
        if pred_norm == target_norm:
            exact_found = True
        # Fuzzy match
        if pred_norm in target_norm or target_norm in pred_norm:
            fuzzy_found = True
    return exact_found, fuzzy_found


def calculate_topk_metrics(
    results: list[dict[str, str | list[str]]],
    topk_list: list[int] = [1, 3, 5],
    fuzzy_match: bool = False,
) -> dict[str, float]:
    """
    Calculate top-k accuracy and fuzzy accuracy for LLM performance.
    Each result's 'prediction' should be a list of top-k predictions.
    """
    max_valid_k = len(results[0]["prediction"])
    topk_list = [k for k in topk_list if k <= max_valid_k]
    metrics = {}
    n = len(results)
    for k in topk_list:
        exact_hits = 0
        fuzzy_hits = 0
        for r in results:
            preds = r["prediction"][:k] if isinstance(r["prediction"], list) else [r["prediction"]]
            target = r["word"]
            ex, fz = is_correct_topk(preds, target, fuzzy_match)
            if ex:
                exact_hits += 1
            if fz:
                fuzzy_hits += 1
        metrics[f"accuracy@{k}"] = exact_hits / n if n else 0.0
        metrics[f"fuzzy_accuracy@{k}"] = fuzzy_hits / n if n else 0.0
    return metrics


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
