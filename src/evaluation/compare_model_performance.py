import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results')

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
        f for f in os.listdir(results_dir)
        if f.endswith('_evaluation_results.json')
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
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    metrics = data.get('overall_metrics', {})
    metrics['model'] = model_name
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
        model_name = fname.replace('_evaluation_results.json', '')
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
    Plot a bar chart comparing model performance across selected metrics.

    Args:
        df (pd.DataFrame): DataFrame containing model metrics.
        metrics (list of str, optional): Metrics to plot. Defaults to all except 'model' and 'num_samples'.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    if metrics is None:
        # Default: plot all metrics except 'model' and 'num_samples'
        metrics = [col for col in df.columns if col not in ('model', 'num_samples') and df[col].dtype != object]
    df_melt = df.melt(id_vars=['model'], value_vars=metrics, var_name='metric', value_name='score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='metric', y='score', hue='model')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Model')
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
    print(df)
    output_csv = os.path.join(RESULTS_DIR, 'model_performance_comparison.csv')
    save_metrics_csv(df, output_csv)
    plot_model_performance(df)

if __name__ == '__main__':
    main()