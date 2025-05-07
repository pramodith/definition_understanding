import json
import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")


def main():
    rows = []
    # Find all *_evaluation_results.json files in the results directory
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith("_evaluation_results.json"):
            model_name = fname.replace("_evaluation_results.json", "")
            file_path = os.path.join(RESULTS_DIR, fname)
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("results", []):
                word = entry.get("word", "").strip()
                word_lower = word.lower()
                predictions = [p.strip() for p in entry.get("prediction", [])]
                predictions_lower = [p.lower() for p in predictions]
                if word_lower not in predictions_lower:
                    rows.append(
                        {
                            "term": word,
                            "incorrect_predictions": ", ".join(predictions),
                            "definition": entry.get("definition", ""),
                            "part_of_speech": entry.get("part_of_speech", ""),
                            "model": model_name,
                            "judge_llm_prediction": entry.get(
                                "judge_llm_prediction", ""
                            ),
                        }
                    )
    df = pd.DataFrame(
        rows,
        columns=[
            "term",
            "incorrect_predictions",
            "definition",
            "part_of_speech",
            "model",
            "judge_llm_prediction",
        ],
    )

    # Save as CSV
    output_path = os.path.join(RESULTS_DIR, "no_exact_match_terms.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved DataFrame to {output_path}")


if __name__ == "__main__":
    main()
