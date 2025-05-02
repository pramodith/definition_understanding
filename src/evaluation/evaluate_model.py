"""
Script to evaluate LLM performance on word definition understanding.

This script loads a dataset of word definitions, queries an LLM to predict
the corresponding words, and evaluates the model's performance.
"""

import argparse
import asyncio
import json
import os

import pandas as pd

from evaluation.metrics import (
    analyze_results_by_category,
    calculate_metrics,
    calculate_topk_metrics,
    save_evaluation_results,
)
from models.model_factory import get_model


def create_prompt(definition: str, part_of_speech: str | None = None) -> str:
    """
    Create a prompt for the LLM to predict a word from its definition.

    Args:
        definition: The definition of the word
        part_of_speech: The part of speech of the word (optional)

    Returns:
        Prompt string
    """

    system_prompt = (
        "You are a word prediction model."
        "You will be given a definition and the part of speech of the word (if available)."
        "You must predict the word being defined."
    )

    pos_info = f" {part_of_speech}" if part_of_speech else "NA"

    instructions = (
        "Respond with just the word and no additional text."
        "# Examples:\n"
        "Definition: A custom-made or tailored item.\nPart of speech: adjective\n"
        "Bespoke\n"
        "Definition: To rage in excess of.\nPart of speech: verb\n"
        "Outrage\n"
    )

    system_prompt = system_prompt + "\n\n" + instructions
    user_query = f"Definition: {definition}\nPart of speech: {pos_info}\n"
    prompt_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    return prompt_message


async def evaluate_model_async(
    model_name: str,
    judgellm_model_name: str,
    dataset_path: str,
    output_dir: str,
    num_samples: int | None = None,
    verbose: bool = False,
    batch_size: int = 5,
    top_logprobs: int | None = None,
) -> dict[str, float]:
    """
    Evaluate an LLM's performance on word definition understanding using async batch processing.

    Args:
        model_name: Name of the LLM to evaluate
        judgellm_model_name: Name of the Judgellm model to evaluate
        dataset_path: Path to the dataset of word definitions
        output_dir: Directory to save evaluation results
        num_samples: Number of samples to evaluate (None for all)
        include_synonyms: Whether to consider synonyms as correct answers
        fuzzy_match: Whether to allow fuzzy matching
        verbose: Whether to print detailed information
        batch_size: Number of prompts to process in each batch
        top_logprobs: Number of top log-probabilities to return

    Returns:
        Dictionary of evaluation metrics
    """
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Sample if requested
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)

    # Get model
    model = get_model(model_name, logprobs=True, top_logprobs=top_logprobs)
    judgellm_model = get_model(judgellm_model_name)

    print(
        f"Evaluating {model_name} on {len(df)} samples using async batch processing..."
    )

    # Prepare data for batch processing
    words = []
    definitions = []
    parts_of_speech = []
    all_synonyms = []
    prompts_messages = []

    # Process each row to extract data and create prompts
    for _, row in df.iterrows():
        word = row["word"]
        definition = row["definition"]
        part_of_speech = row.get("part_of_speech", None)

        # Get synonyms if available
        synonyms = []
        for syn_col in ["synonyms", "wordnet_synonyms", "all_synonyms"]:
            if syn_col in row and row[syn_col]:
                try:
                    if isinstance(row[syn_col], str):
                        # Parse JSON string if needed
                        if row[syn_col].startswith("["):
                            syn_list = json.loads(row[syn_col])
                            synonyms.extend(syn_list)
                        else:
                            synonyms.append(row[syn_col])
                    elif isinstance(row[syn_col], list):
                        synonyms.extend(row[syn_col])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Create prompt
        prompt_message = create_prompt(definition, part_of_speech)

        # Store all data
        words.append(word)
        definitions.append(definition)
        parts_of_speech.append(part_of_speech if pd.notna(part_of_speech) else "")
        all_synonyms.append(synonyms)
        prompts_messages.append(prompt_message)

    # Process all prompts in batches asynchronously
    responses = await model.abatch_generate(prompts_messages, batch_size=batch_size)

    # Process results
    results = []
    for i, prediction_list in enumerate(responses):
        # prediction_list is already a list of top-k predictions
        result = {
            "word": words[i],
            "definition": definitions[i],
            "part_of_speech": parts_of_speech[i],
            "synonyms": all_synonyms[i],
            "prediction": prediction_list,
            "full_response": prediction_list,  # Optionally store the full list
        }
        results.append(result)

        if verbose:
            print(f"Word: {words[i]}")
            print(f"Definition: {definitions[i]}")
            print(f"Predictions: {prediction_list}")
            # Show if any top-k prediction matches
            correct = any(
                p.lower() == words[i].lower()
                or p.lower() in [s.lower() for s in all_synonyms[i]]
                for p in prediction_list
            )
            print(f"Correct (any top-k): {correct}")
            print("-" * 50)

    # Calculate metrics
    metrics = calculate_metrics(results, judgellm_model)
    topk_metrics = calculate_topk_metrics(
        results, topk_list=[1, 3, 5], judgellm_model=judgellm_model
    )
    metrics.update(topk_metrics)

    # Analyze results by category
    category_metrics = analyze_results_by_category(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_name.split("/")[-1]
    output_file = os.path.join(output_dir, f"{model_name}_evaluation_results.json")
    save_evaluation_results(results, metrics, category_metrics, output_file)

    # Print summary
    print(f"\nEvaluation results for {model_name}:")
    print(f"Exact match accuracy: {metrics['exact_accuracy']:.4f}")
    print(f"Accuracy with synonyms: {metrics['synonym_accuracy']:.4f}")
    print(f"Judgellm accuracy: {metrics['judgellm_accuracy']:.4f}")
    print(f"Number of samples: {metrics['num_samples']}")
    for k in [1, 3, 5]:
        print(f"accuracy@{k}: {metrics.get(f'accuracy@{k}', 0):.4f}")
        print(f"fuzzy_accuracy@{k}: {metrics.get(f'fuzzy_accuracy@{k}', 0):.4f}")
        print(f"judgellm_accuracy@{k}: {metrics.get(f'judgellm_accuracy@{k}', 0):.4f}")

    return metrics


def evaluate_model(
    model_name: str,
    judgellm_model_name: str,
    dataset_path: str,
    output_dir: str,
    num_samples: int | None = None,
    verbose: bool = False,
    batch_size: int = 20,
    top_logprobs: int | None = None,
) -> dict[str, float]:
    """
    Evaluate an LLM's performance on word definition understanding.
    This is a wrapper around the async version that runs the event loop.

    Args:
        model_name: Name of the LLM to evaluate
        judgellm_model_name: Name of the Judgellm model to evaluate
        dataset_path: Path to the dataset of word definitions
        output_dir: Directory to save evaluation results
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed information
        batch_size: Number of prompts to process in each batch
        top_logprobs: Number of top log-probabilities to return

    Returns:
        Dictionary of evaluation metrics.........
    """
    # Run the async evaluation in the event loop
    return asyncio.run(
        evaluate_model_async(
            model_name=model_name,
            judgellm_model_name=judgellm_model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_samples=num_samples,
            verbose=verbose,
            batch_size=batch_size,
            top_logprobs=top_logprobs,
        )
    )


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM performance on word definition understanding"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Name of the model to evaluate",
        default="meta-llama/Llama-3.2-3B-Instruct-Turbo",
    )
    parser.add_argument(
        "--judgellm_model_name",
        type=str,
        required=False,
        help="Name of the Judgellm model to evaluate",
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to the dataset of word definitions",
        default="data/processed_dictionary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Directory to save evaluation results",
        default="results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        help="Number of samples to evaluate (None for all)",
        default=10,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of prompts to process in each batch",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=5,
        help="Number of top log-probabilities to return",
    )

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        judgellm_model_name=args.judgellm_model_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        verbose=args.verbose,
        batch_size=args.batch_size,
        top_logprobs=args.top_logprobs,
    )


if __name__ == "__main__":
    main()
