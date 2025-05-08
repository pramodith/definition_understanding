"""
Metrics module for evaluating LLM performance on word definition tasks.

This module provides functions for calculating various metrics to evaluate
how well LLMs understand word definitions, including handling synonyms.
"""

import json
import re

import pandas as pd
from tqdm import tqdm

from models.base_model import LLMModel


def normalize_word(word: str) -> str:
    """
    Normalize a word for comparison.

    Args:
        word: Word to normalize

    Returns:
        Normalized word
    """
    stop_tokens = ["<|eot_id|>", "</s>"]
    # Convert to lowercase
    word = word.lower()
    for stop_token in stop_tokens:
        word = word.replace(stop_token, "")
    # Remove punctuation and special characters
    word = re.sub(r"[^\w\s]", "", word)

    # Remove extra whitespace
    word = re.sub(r"\s+", " ", word).strip()

    return word


def is_correct_answer(
    result: dict,
    judgellm_model: LLMModel | None = None,
) -> dict:
    """
    Check if the prediction is correct, considering synonyms and JudgeLLM.

    Args:
        result: Dictionary containing keys 'prediction', 'word', 'definition', 'synonyms', etc.
        judgellm_model: Optional LLMModel instance for JudgeLLM

    Returns:
        Dictionary with keys 'exact', 'synonym', 'fuzzy', 'judgellm'
    """
    prediction = result.get("prediction", "")
    target = result.get("word", "")
    definition = result.get("definition", "")
    synonyms = result.get("synonyms", [])
    preds = prediction if isinstance(prediction, list) else [prediction]
    target_norm = normalize_word(target)
    normalized_synonyms = [normalize_word(syn) for syn in synonyms] if synonyms else []
    is_correct = {"exact": False, "synonym": False, "fuzzy": False, "judgellm": False}
    for pred in preds:
        pred_norm = normalize_word(pred)
        # Check exact match
        if pred_norm == target_norm:
            is_correct["exact"] = True
            is_correct["synonym"] = True
            is_correct["fuzzy"] = True
            is_correct["judgellm"] = True
        # Check synonyms if provided
        if (
            not is_correct["synonym"]
            and normalized_synonyms
            and pred_norm in normalized_synonyms
        ):
            is_correct["synonym"] = is_correct["exact"] or True
        # Check fuzzy match
        if not is_correct["fuzzy"]:
            if (pred_norm in target_norm) or (target_norm in pred_norm):
                is_correct["fuzzy"] = is_correct["exact"] or True
    # If none are True, call JudgeLLM if provided
    if not is_correct["exact"] and judgellm_model:
        for pred in preds:
            is_equiv = judge_llm_equivalence(
                normalize_word(pred), target_norm, definition, judgellm_model
            )
            if is_equiv:
                is_correct["judgellm"] = True
                break
    return is_correct



def judge_llm_equivalence(
    prediction: str, target: str, definition: str, llm_model: LLMModel
) -> bool:
    """
    Uses an LLM to judge if two words are equivalent or synonymous (ignoring tense, plurality, etc).
    Returns True if the LLM says they are equivalent or synonymous.
    """
    examples = """## Examples
    Definition: Bleeding from the nose, usually due to ruptured blood vessels in the nasal mucosa.
    Are the words 'Epistaxis' and 'Nosebleed' the same or synonymous? Yes
    Definition: The largest part of the brain, responsible for higher brain functions like thought, action, and sensory processing.
    Are the words 'Cerebrum' and 'Forebrain' the same or synonymous? Yes
    Definition: An elevated body temperature, often due to infection or illness.
    Are the words 'Fever' and 'diarrhea' the same or synonymous? No
    Definition: a single-stranded RNA molecule that carries genetic information"\
        "and from the DNA in the cell's nucleus to the cytoplasm, where proteins are synthesized"
    Are the words 'mRNA' and 'Messenger RNA' the same or synonymous? Yes
    Definition: beat or sound with a strong, regular rhythm; pulsate steadily.
    Are the words 'throb' and 'throbbing' the same or synonymous? Yes
    """
    prompt = [
        {
            "role": "system",
            "content": "You are an expert linguist. "
            "You will judge if two words are the same or synonymous given a definition. Respond with a single word: 'Yes' or 'No'.\n\n"
            + examples,
        },
        {
            "role": "user",
            "content": f"Definition: {definition}"
            f" Are the words '{prediction}' and '{target}' the same or synonymous?",
        },
    ]
    try:
        response = llm_model.generate(prompt)[0]
        # Accept 'yes' if it's in the first token or anywhere in the response
        return "yes" in response.lower()
    except Exception as e:
        print(f"JudgeLLM error: {e}")
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
    judgellm_model: LLMModel | None = None,
) -> dict[str, float]:
    """
    Calculate evaluation metrics for LLM performance.

    Args:
        results: List of dictionaries with keys 'word', 'definition', 'prediction', and optionally 'synonyms'

    Returns:
        Dictionary of metrics
    """
    is_correct = []
    for i in tqdm(range(len(results)), desc="Judging ..."):
        is_correct.append(is_correct_answer(results[i], judgellm_model=judgellm_model))
        results[i]["judge_llm_prediction"] = is_correct[i]["judgellm"]

    exact_accuracy = sum(is_correct[i]["exact"] for i in range(len(is_correct))) / len(
        is_correct
    )
    synonym_accuracy = sum(
        is_correct[i]["synonym"] for i in range(len(is_correct))
    ) / len(is_correct)
    fuzzy_accuracy = sum(is_correct[i]["fuzzy"] for i in range(len(is_correct))) / len(
        is_correct
    )
    judgellm_accuracy = sum(
        is_correct[i]["judgellm"] for i in range(len(is_correct))
    ) / len(is_correct)

    return {
        "exact_accuracy": exact_accuracy,
        "synonym_accuracy": synonym_accuracy,
        "fuzzy_accuracy": fuzzy_accuracy,
        "judgellm_accuracy": judgellm_accuracy,
        "num_samples": len(results),
    }, results


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
    definition: str,
    judgellm_model: LLMModel | None = None,
) -> tuple[bool, bool, bool]:
    """
    Check if the target is in the top-k predictions (exact and fuzzy).
    Args:
        predictions: List of top-k predicted tokens/words
        target: The target word
        definition: The definition of the target word
        judgellm_model: Optional JudgeLLM model for fuzzy matching
    Returns:
        exact_found: Whether the target was found in the top-k predictions (exact match)
        fuzzy_found: Whether the target was found in the top-k predictions (fuzzy match)
        judgellm_found: Whether the target was found in the top-k predictions (JudgeLLM match)
    """
    # Normalize target and synonyms
    target_norm = normalize_word(target)
    exact_found = False
    fuzzy_found = False
    judgellm_found = False
    for pred in predictions:
        pred_norm = normalize_word(pred)
        # Exact match or synonym
        if pred_norm == target_norm:
            exact_found = True
            judgellm_found = True
        # Fuzzy match
        if pred_norm in target_norm or target_norm in pred_norm:
            fuzzy_found = True
        # JudgeLLM
        if judgellm_model and not exact_found:
            if judge_llm_equivalence(
                pred_norm, target_norm, definition, judgellm_model
            ):
                judgellm_found = True

    return exact_found, fuzzy_found, judgellm_found


def calculate_topk_metrics(
    results: list[dict[str, str | list[str]]],
    topk_list: list[int] = [1, 3, 5],
    judgellm_model: LLMModel | None = None,
) -> dict[str, float]:
    """
    Calculate top-k accuracy and fuzzy accuracy for LLM performance.
    Each result's 'prediction' should be a list of top-k predictions.
    Args:
        results: List of dictionaries with keys 'word', 'definition', 'prediction', and optionally 'synonyms'
        topk_list: List of top-k values to calculate accuracy for
        judgellm_model: Optional JudgeLLM model for fuzzy matching
    Returns:
        Dictionary of metrics
    """
    max_valid_k = len(results[0]["prediction"])
    topk_list = [k for k in topk_list if k <= max_valid_k]
    metrics = {}
    n = len(results)
    for k in topk_list:
        exact_hits = 0
        judgellm_hits = 0
        fuzzy_hits = 0
        for r in results:
            preds = (
                r["prediction"][:k]
                if isinstance(r["prediction"], list)
                else [r["prediction"]]
            )
            target = r["word"]
            definition = r["definition"]
            ex, fz, judgellm = is_correct_topk(
                preds, target, definition, judgellm_model
            )
            if ex:
                exact_hits += 1
            if fz:
                fuzzy_hits += 1
            if judgellm:
                judgellm_hits += 1
        metrics[f"accuracy@{k}"] = exact_hits / n if n else 0.0
        metrics[f"fuzzy_accuracy@{k}"] = fuzzy_hits / n if n else 0.0
        metrics[f"judgellm_accuracy@{k}"] = judgellm_hits / n if n else 0.0
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
