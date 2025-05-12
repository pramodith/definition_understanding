"""
Metrics module for evaluating LLM performance on word definition tasks.

This module provides functions for calculating various metrics to evaluate
how well LLMs understand word definitions, including handling synonyms.
"""

import json
import re
from typing import Any

import pandas as pd

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
) -> dict:
    """
    Check if the prediction is correct, considering synonyms and JudgeLLM.
    Args:
        result: Dictionary containing keys 'prediction', 'word', 'synonyms', etc.
    Returns:
        Dictionary with keys 'exact', 'synonym', 'fuzzy', 'judgellm'
    """
    prediction = result.get("prediction", "")
    target = result.get("word", "")
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
    return is_correct


async def ajudge_llm_equivalence(
    pairs: list[dict[str, Any]], llm_model: LLMModel
) -> list[bool]:
    """
    Async batch version of judge_llm_equivalence using abatch_generate.
    Args:
        pairs: List of dicts with keys 'prediction', 'target', 'definition'.
        llm_model: LLMModel instance.
    Returns:
        List of bools indicating equivalence for each pair.
    """
    examples = """## Examples
    Definition: Bleeding from the nose, usually due to ruptured blood vessels in the nasal mucosa.
    Are the words 'Epistaxis' and 'Nosebleed' the same or synonymous? Yes
    Definition: The largest part of the brain, responsible for higher brain functions like thought, action, and sensory processing.
    Are the words 'Cerebrum' and 'Forebrain' the same or synonymous? Yes
    Definition: An elevated body temperature, often due to infection or illness.
    Are the words 'Fever' and 'diarrhea' the same or synonymous? No
    Definition: a single-stranded RNA molecule that carries genetic information"
        "and from the DNA in the cell's nucleus to the cytoplasm, where proteins are synthesized"
    Are the words 'mRNA' and 'Messenger RNA' the same or synonymous? Yes
    Definition: beat or sound with a strong, regular rhythm; pulsate steadily.
    Are the words 'throb' and 'throbbing' the same or synonymous? Yes
    """
    prompts = []
    for pair in pairs:
        prompt = [
            {
                "role": "system",
                "content": "You are an expert linguist. "
                "You will judge if two words are the same or synonymous given a definition. Respond with a single word: 'Yes' or 'No'.\n\n"
                + examples,
            },
            {
                "role": "user",
                "content": f"Definition: {pair['definition']}"
                f" Are the words '{pair['prediction']}' and '{pair['target']}' the same or synonymous?",
            },
        ]
        prompts.append(prompt)
    responses = await llm_model.abatch_generate(prompts)

    # abatch_generate returns List[List[str]], so flatten and check for 'yes'
    results = []
    for resp_list in responses:
        # get the top-1 response
        resp = resp_list[0] if resp_list else ""
        results.append("yes" in resp.lower())
    return results


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


async def calculate_metrics(
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
    exact_accuracy = 0
    synonym_accuracy = 0
    fuzzy_accuracy = 0
    judgellm_accuracy = 0

    # Rule based metrics
    for i in range(len(results)):
        if "exact" not in results[i]:
            judgement = is_correct_answer(results[i])
            results[i]["exact"] = judgement["exact"]
            results[i]["synonym"] = judgement["synonym"]
            results[i]["fuzzy"] = judgement["fuzzy"]
            results[i]["judgellm"] = judgement["judgellm"]

        exact_accuracy += results[i]["exact"]
        synonym_accuracy += results[i]["synonym"]
        fuzzy_accuracy += results[i]["fuzzy"]

    # JudgeLLM metrics
    if judgellm_model:
        query_params = []
        for i in range(len(results)):
            if not results[i]["exact"]:
                query_params.append(
                    {
                        "prediction": results[i]["prediction"][0],
                        "target": results[i]["word"],
                        "definition": results[i]["definition"],
                        "index": i,
                    }
                )

        judgellm_results = await ajudge_llm_equivalence(query_params, judgellm_model)
        for i in range(len(judgellm_results)):
            results[query_params[i]["index"]]["judgellm"] = judgellm_results[i]

        judgellm_accuracy = sum(
            results[i]["judgellm"] for i in range(len(results))
        )

    return {
        "exact_accuracy": exact_accuracy / len(results),
        "synonym_accuracy": synonym_accuracy / len(results),
        "fuzzy_accuracy": fuzzy_accuracy / len(results),
        "judgellm_accuracy": judgellm_accuracy / len(results),
        "num_samples": len(results),
    }, results


async def analyze_results_by_category(
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
                pos_metrics[pos], _ = await calculate_metrics(pos_results)

    # Calculate metrics by word length category
    length_metrics = {}
    for category in df["length_category"].unique():
        if pd.isna(category):
            continue

        category_results = df[df["length_category"] == category].to_dict("records")
        if category_results:
            length_metrics[str(category)] = await calculate_metrics(category_results)

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
