"""
Module for preprocessing dictionary data and handling synonyms.

This module provides functions for cleaning and preprocessing dictionary data,
as well as for handling synonyms using NLTK's WordNet.
"""

import os
import re

import pandas as pd


def clean_definition(definition: str) -> str:
    """
    Clean a definition by removing special characters and normalizing whitespace.

    Args:
        definition: The definition to clean

    Returns:
        Cleaned definition
    """
    # Remove any references like "[1]" often found in dictionary entries
    definition = re.sub(r"\[\d+\]", "", definition)

    # Remove any parenthetical clarifications if needed
    definition = re.sub(r"\([^)]*\)", "", definition)

    # Normalize whitespace
    definition = re.sub(r"\s+", " ", definition).strip()

    return definition


def create_evaluation_dataset(
    input_file: str,
    output_file: str,
    min_word_length: int = 3,
    max_word_length: int = 15,
    clean_definitions: bool = True,
) -> None:
    """
    Create evaluation datasets from processed dictionary data.

    Args:
        input_file: Path to the processed dictionary data file
        output_file: Path to save the evaluation dataset
        min_word_length: Minimum length of words to include
        max_word_length: Maximum length of words to include
        clean_definitions: Whether to clean definitions
    """
    # Load the processed data
    df = pd.read_csv(input_file)

    # Filter by word length
    df = df[df["word"].str.len().between(min_word_length, max_word_length)]

    # Clean definitions if requested
    if clean_definitions:
        df["definition"] = df["definition"].apply(clean_definition)

    # Save the evaluation dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(
        f"Evaluation dataset created with {len(df)} entries and saved to {output_file}"
    )


if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )
    input_file = os.path.join(data_dir, "processed_dictionary.csv")
    output_file = os.path.join(data_dir, "evaluation_dataset.csv")

    create_evaluation_dataset(input_file, output_file)
