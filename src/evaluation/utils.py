from dataclasses import dataclass


@dataclass
class EvaluationModels:
    model_name: str
    judgellm_model_name: str
    is_local: bool


def is_correct_topk(
    predictions: list[str],
    target: str,
    definition: str,
    judgellm_model: LLMModel | None = None,
) -> tuple[bool, bool, bool]:
    """
    Check if the target is in the top-k predictions (exact and fuzzy). Unused right now.
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
    Unused right now.
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
