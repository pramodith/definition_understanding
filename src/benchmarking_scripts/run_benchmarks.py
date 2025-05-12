"""
Script to run benchmarks for all models specified in model_list in evaluate_model.py.
"""

from evaluation.evaluate_model import evaluate_model
from evaluation.utils import EvaluationModels

model_list = [
    EvaluationModels(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        judgellm_model_name="gpt-4.1-2025-04-14",
        is_local=True,
    ),
    EvaluationModels(
        model_name="phi-4",
        judgellm_model_name="gpt-4.1-2025-04-14",
        is_local=True,
    ),
    EvaluationModels(
        model_name="Qwen/Qwen3-8B",
        judgellm_model_name="gpt-4.1-2025-04-14",
        is_local=True,
    ),
    # EvaluationModels(
    #         model_name="gpt-4.1-nano",
    #         judgellm_model_name="gpt-4.1-2025-04-14",
    #         is_local=False,
    #     ),
    # EvaluationModels(
    #     model_name="gpt-4.1-mini",
    #     judgellm_model_name="gpt-4.1-2025-04-14",
    #     is_local=False,
    # ),
    # EvaluationModels(
    #     model_name="gpt-4.1",
    #     judgellm_model_name="gpt-4.1-2025-04-14",
    #     is_local=False,
    # ),
    # EvaluationModels(
    #     model_name="claude-3-7-sonnet-20250219",
    #     judgellm_model_name="gpt-4.1-2025-04-14",
    #     is_local=False,
    # ),
    # EvaluationModels(
    #     model_name="gemini-2.0-flash",
    #     judgellm_model_name="gpt-4.1-2025-04-14",
    #     is_local=False,
    # ),
]

# Default args (can be customized)
default_args = {
    "dataset_path": "data/processed_dictionary.csv",
    "output_dir": "results",
    "num_samples": None,  # Set to an int for a subset
    "verbose": False,
}

def run_all_benchmarks():
    for model in model_list:
        print(f"\nEvaluating model: {model.model_name}")
        evaluate_model(
            model_name=model.model_name,
            judgellm_model_name=model.judgellm_model_name,
            dataset_path=default_args["dataset_path"],
            output_dir=default_args["output_dir"],
            num_samples=default_args["num_samples"],
            verbose=default_args["verbose"],
            is_local=model.is_local,
        )

if __name__ == "__main__":
    run_all_benchmarks()
