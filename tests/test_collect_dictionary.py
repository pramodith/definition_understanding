"""
Tests for the collect_dictionary module.

This module contains tests for the dictionary data collection functions.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from data_processing.collect_dictionary import (
    fetch_definition_from_free_dictionary,
    get_word_list,
    get_wikipedia_medical_glossary_word_list,
    process_dictionary_data,
    fetch_medical_definitions,
    parse_wordsapi_entry,
    parse_medical_entry,
)


def test_get_word_list_from_file():
    """Test getting a word list from a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary word list file
        word_list_file = os.path.join(temp_dir, "word_list.txt")
        test_words = ["apple", "banana", "cat", "dog", "elephant"]

        with open(word_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(test_words))

        # Test reading from the file
        result = get_word_list(file_path=word_list_file)

        # Check that the words were read correctly
        assert result == test_words


@patch("requests.get")
def test_get_word_list_from_api(mock_get):
    """Test getting a word list from the Datamuse API."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"word": "apple", "score": 100},
        {"word": "banana", "score": 90},
        {"word": "cat", "score": 80},
    ]
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the default word list file path
        with patch(
            "src.data_processing.collect_dictionary.DEFAULT_WORD_LIST_FILE",
            os.path.join(temp_dir, "word_list.txt"),
        ):
            # Test fetching from the API
            result = get_word_list(file_path=None, num_words=3)

            # Check that the API was called
            mock_get.assert_called()

            # Check that the words were fetched correctly
            assert result == ["apple", "banana", "cat"]

            # Check that the word list was saved to a file
            # assert os.path.exists(os.path.join(temp_dir, "word_list.txt"))


@patch("requests.get")
def test_fetch_definition_from_free_dictionary(mock_get):
    """Test fetching a definition from the Free Dictionary API."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "word": "apple",
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definitions": [
                        {
                            "definition": "A round fruit with red or green skin and a white inside",
                            "synonyms": ["fruit"],
                        }
                    ],
                }
            ],
        }
    ]
    mock_get.return_value = mock_response

    # Test fetching a definition
    result = fetch_definition_from_free_dictionary("apple")

    # Check that the API was called with the correct URL
    mock_get.assert_called_with("https://api.dictionaryapi.dev/api/v2/entries/en/apple")

    # Check that the definition was fetched correctly
    assert result == mock_response.json.return_value


def test_process_dictionary_data():
    """Test processing raw dictionary data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary input file
        input_file = os.path.join(temp_dir, "dictionary_data.json")
        output_file = os.path.join(temp_dir, "processed_dictionary.csv")

        # Create dummy data
        raw_data = {
            "apple": [
                {
                    "word": "apple",
                    "meanings": [
                        {
                            "partOfSpeech": "noun",
                            "definitions": [
                                {
                                    "definition": "A round fruit with red or green skin and a white inside",
                                    "synonyms": ["fruit"],
                                    "antonyms": [],
                                }
                            ],
                        }
                    ],
                }
            ],
            "banana": [
                {
                    "word": "banana",
                    "meanings": [
                        {
                            "partOfSpeech": "noun",
                            "definitions": [
                                {
                                    "definition": "A long curved fruit with yellow skin",
                                    "synonyms": ["fruit"],
                                    "antonyms": [],
                                }
                            ],
                        }
                    ],
                }
            ],
            "error_word": {"error": "Word not found"},
        }

        # Save the raw data to the input file
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)

        # Test processing the data
        process_dictionary_data(
            input_file=input_file,
            output_file=output_file,
            min_definition_length=5,
            max_definition_length=100,
        )

        # Check that the output file exists
        assert os.path.exists(output_file)

        # Read the processed data
        df = pd.read_csv(output_file)

        # Check that the processed data has the expected structure
        assert "word" in df.columns
        assert "definition" in df.columns
        assert "part_of_speech" in df.columns
        assert "synonyms" in df.columns
        assert "antonyms" in df.columns

        # Check that the words were processed correctly
        assert "apple" in df["word"].values
        assert "banana" in df["word"].values

        # Check that the error word was skipped
        assert "error_word" not in df["word"].values

        # Check that the definitions were processed correctly
        apple_def = df[df["word"] == "apple"]["definition"].iloc[0]
        assert apple_def == "A round fruit with red or green skin and a white inside"

        banana_def = df[df["word"] == "banana"]["definition"].iloc[0]
        assert banana_def == "A long curved fruit with yellow skin"


@patch("requests.get")
def test_fetch_medical_definitions(mock_get):
    """Test fetching a definition from the Medical Dictionary API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "meta": {"id": "aspirin"},
            "shortdef": ["A medication used to reduce pain, fever, or inflammation."],
        }
    ]
    mock_get.return_value = mock_response

    result = fetch_medical_definitions("aspirin")
    assert result["definition"] == "A medication used to reduce pain, fever, or inflammation."
    assert result["synonyms"] == []
    mock_response.json.return_value = [
        {
            "fl": "noun",
            "shortdef": ["A medication used to reduce pain, fever, or inflammation."],
            "meta": {"syns": ["painkiller"], "stems": ["painfree", "abs"]}
        }
    ]
    mock_get.return_value = mock_response
    result = fetch_medical_definitions("aspirin")
    assert result["synonyms"] == ["painkiller", "painfree", "abs"] and result["part_of_speech"] == "noun"

@patch("requests.get")
def test_get_wikipedia_medical_glossary_word_list(mock_get):
    """Test getting medical glossary word list from Wikipedia."""
    html = '''<div class="mw-parser-output"><dt>Aspirin</dt><dt>Ibuprofen</dt></div>'''
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html
    mock_get.return_value = mock_response
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "medical_word_list.txt")
        word_list = get_wikipedia_medical_glossary_word_list(output_file=output_file)
        assert "Aspirin" in word_list
        assert "Ibuprofen" in word_list
        assert os.path.exists(output_file)

def test_parse_medical_entry():
    word = "aspirin"
    data = {
            "definition": "A medication used to reduce pain, fever, or inflammation.",
            "part_of_speech": "noun",
            "synonyms": ["painkiller", "painfree", "abs"]
        }
    entries = parse_medical_entry(word, data)
    assert isinstance(entries, list)
    assert entries[0]["word"] == "aspirin"
    assert entries[0]["definition"] == "A medication used to reduce pain, fever, or inflammation."
    assert entries[0]["part_of_speech"] == "noun"
    assert entries[0]["synonyms"] == ["painkiller", "painfree", "abs"]


def test_parse_wordsapi_entry():
    word = "aspirin"
    data = {
        "word": "aspirin",
        "definitions": [
            {"definition": "A drug used to reduce pain and fever."}
        ],
    }
    entries = parse_wordsapi_entry(word, data)
    assert isinstance(entries, list)
    assert entries[0]["word"] == "aspirin"
    assert "pain" in entries[0]["definition"]

