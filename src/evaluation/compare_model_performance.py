import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")


# List all *_evaluation_results.json files in the results directory
def get_result_files(results_dir):
    """
    List all evaluation result JSON files in the specified directory.

    Args:
        results_dir (str): Path to the directory containing result files.

    Returns:
        list of str: Filenames ending with '_evaluation_results.json'.
    """
    return [
        f for f in os.listdir(results_dir) if f.endswith("_evaluation_results.json")
    ]


def load_metrics_from_file(filepath, model_name):
    """
    Load overall metrics from a JSON result file and add the model name.

    Args:
        filepath (str): Path to the JSON file.
        model_name (str): Name of the model (used for identification).

    Returns:
        dict: Dictionary of overall metrics with model name included.
    """
    with open(filepath, encoding="utf-8") as f:
        print(f"Reading results of {model_name}")
        data = json.load(f)
    metrics = data.get("overall_metrics", {})
    metrics["model"] = model_name
    return metrics


def aggregate_model_metrics(results_dir):
    """
    Aggregate overall metrics from all model result files in the directory into a DataFrame.

    Args:
        results_dir (str): Path to the directory containing result files.

    Returns:
        pd.DataFrame: DataFrame with each row representing a model's metrics.
    """
    files = get_result_files(results_dir)
    all_metrics = []
    for fname in files:
        model_name = fname.replace("_evaluation_results.json", "")
        filepath = os.path.join(results_dir, fname)
        metrics = load_metrics_from_file(filepath, model_name)
        all_metrics.append(metrics)
    df = pd.DataFrame(all_metrics)
    return df


def save_metrics_csv(df, output_path):
    """
    Save the aggregated metrics DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing model metrics.
        output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Saved metrics comparison CSV to {output_path}")


def plot_model_performance(df, metrics=None, save_path=None):
    """
    Plot separate bar chart subplots comparing model performance for each selected metric.

    Args:
        df (pd.DataFrame): DataFrame containing model metrics.
        metrics (list of str, optional): Metrics to plot. Defaults to common accuracy metrics.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    import math

    # Define default metrics if not provided
    if metrics is None:
        metrics = [
            m
            for m in [
                "exact_accuracy",
                "judgellm_accuracy",
                "fuzzy_accuracy",
                "synonym_accuracy",
            ]
            if m in df.columns
        ]
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(df["model"], df[metric], color="skyblue")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved model performance plots to {save_path}")
    else:
        plt.show()

    df_melt = df.melt(
        id_vars=["model"], value_vars=metrics, var_name="metric", value_name="score"
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x="metric", y="score", hue="model")
    plt.title("Definition Understanding Model Performance")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Model")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    """
    Main function to aggregate model results, save the comparison CSV, and plot performance.
    """
    df = aggregate_model_metrics(RESULTS_DIR)
    output_csv = os.path.join(RESULTS_DIR, "model_performance_comparison.csv")
    save_metrics_csv(df, output_csv)
    plot_model_performance(
        df, save_path=f"{RESULTS_DIR}/model_performance_comparison.png"
    )


if __name__ == "__main__":
    main()
