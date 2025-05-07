"""
Tests for the preprocess module.

This module contains tests for the preprocess module functions.
"""

import os
import tempfile

import pandas as pd
import pytest

from data_processing.preprocess import clean_definition, create_evaluation_dataset


def test_clean_definition():
    """Test the clean_definition function."""
    # Test removing references
    assert clean_definition("This is a definition [1]") == "This is a definition"

    # Test removing parenthetical clarifications
    assert (
        clean_definition("This is a definition (with clarification)")
        == "This is a definition"
    )

    # Test normalizing whitespace
    assert clean_definition("This   is  a   definition") == "This is a definition"

    # Test combined cases
    assert (
        clean_definition("This   is  a [2] definition (with clarification)")
        == "This is a definition"
    )


def test_create_evaluation_dataset():
    """Test the create_evaluation_dataset function."""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy input CSV file
        input_file = os.path.join(temp_dir, "input.csv")
        output_file = os.path.join(temp_dir, "output.csv")

        # Create dummy data
        data = {
            "word": [
                "apple",
                "banana",
                "cat",
                "supercalifragilisticexpialidocious",
                "a",
            ],
            "definition": [
                "A round fruit with red or green skin and a white inside [1]",
                "A long curved fruit with yellow skin (and sweet flesh)",
                "A   small  domesticated  carnivorous mammal",
                "A made-up word that is very long",
                "The first letter of the alphabet",
            ],
            "part_of_speech": ["noun", "noun", "noun", "adjective", "noun"],
        }

        # Create the input DataFrame and save it
        df = pd.DataFrame(data)
        df.to_csv(input_file, index=False)

        # Test with default parameters
        create_evaluation_dataset(
            input_file=input_file,
            output_file=output_file,
            min_word_length=3,
            max_word_length=15,
            clean_definitions=True,
        )

        # Read the output file
        result_df = pd.read_csv(output_file)

        # Check that the output file exists and has the expected content
        assert os.path.exists(output_file)

        # Check that words with length < min_word_length are filtered out
        assert "a" not in result_df["word"].values

        # Check that words with length > max_word_length are filtered out
        assert "supercalifragilisticexpialidocious" not in result_df["word"].values

        # Check that the remaining words are included
        assert "apple" in result_df["word"].values
        assert "banana" in result_df["word"].values
        assert "cat" in result_df["word"].values

        # Check that definitions are cleaned
        apple_def = result_df[result_df["word"] == "apple"]["definition"].iloc[0]
        assert apple_def == "A round fruit with red or green skin and a white inside"

        banana_def = result_df[result_df["word"] == "banana"]["definition"].iloc[0]
        assert banana_def == "A long curved fruit with yellow skin"

        cat_def = result_df[result_df["word"] == "cat"]["definition"].iloc[0]
        assert cat_def == "A small domesticated carnivorous mammal"

        # Test without cleaning definitions
        create_evaluation_dataset(
            input_file=input_file,
            output_file=output_file,
            min_word_length=3,
            max_word_length=15,
            clean_definitions=False,
        )

        # Read the output file
        result_df = pd.read_csv(output_file)

        # Check that definitions are not cleaned
        banana_def = result_df[result_df["word"] == "banana"]["definition"].iloc[0]
        assert banana_def == "A long curved fruit with yellow skin (and sweet flesh)"

        cat_def = result_df[result_df["word"] == "cat"]["definition"].iloc[0]
        assert cat_def == "A   small  domesticated  carnivorous mammal"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
