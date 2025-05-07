"""
Script to collect dictionary data from various sources.

This module provides functionality to fetch word definitions from online
dictionary APIs and save them in a structured format for further processing.
"""

import argparse
import json
import os
import random
import time
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pandas as pd
import requests
from tqdm import tqdm

load_dotenv()

DICTIONARY_API_KEY = os.getenv("DICTIONARY_API_KEY")

# Constants
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)
DEFAULT_MEDICAL_WORD_LIST_FILE = os.path.join(
    DEFAULT_OUTPUT_DIR, "medical_word_list.txt"
)
DEFAULT_WORD_LIST_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "word_list.txt")
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "dictionary_data.json")
DEFAULT_PROCESSED_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "processed_dictionary.csv")

# Free Dictionary API endpoint
DICTIONARY_API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
MEDICAL_DICTIONARY_API_URL = "https://dictionaryapi.com/api/v3/references/medical/json"

# WordsAPI (alternative, requires API key)
WORDS_API_URL = "https://wordsapiv1.p.rapidapi.com/words/{word}/definitions"
WORDS_API_HEADERS = {
    "X-RapidAPI-Key": "",  # Add your API key here if using WordsAPI
    "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com",
}


def get_wikipedia_medical_glossary_word_list(
    output_file: str = DEFAULT_MEDICAL_WORD_LIST_FILE,
):
    wiki_url = "https://en.wikipedia.org/wiki/Glossary_of_medicine"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(wiki_url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(
            f"Failed to load page {wiki_url} with status code {response.status_code}"
        )

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"class": "mw-parser-output"})

    terms = []
    for tag in content_div.find_all(["dt", "b"]):
        term = tag.get_text(strip=True)
        if term and len(term.split()) < 2 and term[0].isalpha():
            terms.append(term)

    word_list = sorted(set(terms))

    # Save the word list for future use
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(word_list))

    return word_list


def get_word_list(file_path: str | None = None, num_words: int = 1000) -> list[str]:
    """
    Get a list of words to fetch definitions for.

    If file_path is provided, reads words from the file.
    Otherwise, fetches a list of common English words.

    Args:
        file_path: Path to a file containing words (one per line)
        num_words: Number of words to fetch if downloading

    Returns:
        List of words
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # If no file provided, fetch words from an online source
    print(f"Fetching {num_words} common English words...")

    # Using Datamuse API to get common words
    word_list = []

    # We'll generate two random letter combinations and fetch words for each
    while len(word_list) < num_words:
        letter1 = random.choice("abcdefghijklmnopqrstuvwxyz")
        letter2 = random.choice("abcdefghijklmnopqrstuvwxyz")

        # Fetch words for each letter combination
        response = requests.get(
            f"https://api.datamuse.com/words?sp={letter1}*{letter2}&max=10"
        )
        if response.status_code == 200:
            data = response.json()
            word_list.extend([item["word"] for item in data if "word" in item])

    # Save the word list for future use
    os.makedirs(os.path.dirname(DEFAULT_WORD_LIST_FILE), exist_ok=True)
    with open(DEFAULT_WORD_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(word_list))

    return word_list


def fetch_medical_definitions(
    word: str,
    existing_data: dict[str, Any],
    delay: float = 0.5,
    max_words: int | None = None,
) -> dict[str, Any]:
    """
    Fetch medical definitions for a word from the Medical Dictionary API.

    Args:
        word: The word to fetch the definition for
        existing_data: List of existing definitions for the word
        delay: Delay between API requests to avoid rate limiting
        max_words: Maximum number of words to process

    Returns:
        Dictionary containing the word's definitions and metadata
    """
    url = f"{MEDICAL_DICTIONARY_API_URL}/{word}?key={DICTIONARY_API_KEY}"
    existing_data[word] = []
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            for entry in data:
                definition = entry.get("shortdef", [""])[0]
                part_of_speech = entry.get("fl", "")
                synonyms = entry.get("meta", {}).get("syns", [])
                synonyms.extend(entry.get("meta", {}).get("stems", []))
                # Synonyms may appear under 'meta' > 'syns' (not always present)
                if "meta" in entry and "syns" in entry["meta"]:
                    for syn_group in entry["meta"]["syns"]:
                        synonyms.extend(syn_group)

                existing_data[word].append(
                    {
                        "definition": definition,
                        "part_of_speech": part_of_speech,
                        "synonyms": synonyms,
                    }
                )
                if max_words and len(existing_data) >= max_words:
                    break
        else:
            suggested_words = data
            for suggested_word in suggested_words:
                if suggested_word not in existing_data:
                    existing_data = fetch_medical_definitions(
                        suggested_word, existing_data, delay, max_words
                    )
                    time.sleep(delay)
                if max_words and len(existing_data) >= max_words:
                    break
        return existing_data
    else:
        print(f"Failed to fetch {word}: status code {response.status_code}")
        return existing_data


def fetch_definition_from_free_dictionary(word: str) -> dict:
    """
    Fetch definition from the Free Dictionary API.

    Args:
        word: The word to fetch the definition for

    Returns:
        Dictionary containing the word's definitions and metadata
    """
    url = DICTIONARY_API_URL.format(word=word)
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Failed to fetch definition for '{word}'",
            "status_code": response.status_code,
        }


def fetch_definition_from_words_api(word: str) -> dict:
    """
    Fetch definition from WordsAPI (alternative, requires API key).

    Args:
        word: The word to fetch the definition for

    Returns:
        Dictionary containing the word's definitions and metadata
    """
    if not WORDS_API_HEADERS["X-RapidAPI-Key"]:
        return {"error": "WordsAPI key not provided"}

    url = WORDS_API_URL.format(word=word)
    response = requests.get(url, headers=WORDS_API_HEADERS)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Failed to fetch definition for '{word}'",
            "status_code": response.status_code,
        }


def collect_dictionary_data(
    word_list: list[str],
    output_file: str = DEFAULT_OUTPUT_FILE,
    api: str = "free_dictionary",
    max_words: int | None = None,
    delay: float = 0.5,
) -> None:
    """
    Collect dictionary data for a list of words.

    Args:
        word_list: List of words to fetch definitions for
        output_file: Path to save the raw dictionary data
        api: API to use ('free_dictionary' or 'words_api')
        max_words: Maximum number of words to process
        delay: Delay between API requests to avoid rate limiting
    """
    if max_words:
        random.seed(42)
        random.shuffle(word_list)
        word_list = word_list[:max_words]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check if output file exists and load existing data
    existing_data = {}
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}

    # Fetch definitions for words not already in the existing data
    new_words = [word for word in word_list if word not in existing_data]

    if not new_words:
        print(f"All {len(word_list)} words already have definitions in {output_file}")
        return

    print(f"Fetching definitions for {len(new_words)} new words...")
    definition_data = None
    for word in tqdm(new_words):
        # Check if we have reached the maximum number of words
        if max_words and len(existing_data) >= max_words:
            break
        try:
            if api == "free_dictionary":
                definition_data = fetch_definition_from_free_dictionary(word)
            elif api == "medical":
                existing_data = fetch_medical_definitions(word, existing_data, delay)
            else:  # words_api
                definition_data = fetch_definition_from_words_api(word)

            if definition_data:
                existing_data[word] = definition_data
        except Exception as e:
            print(f"Failed to fetch definition for '{word}': {str(e)}")

        # Add delay to avoid rate limiting
        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)

    print(f"Dictionary data collected and saved to {output_file}")


def parse_medical_entry(
    word, data, min_definition_length=50, max_definition_length=200
):
    results = []
    for entry in data:
        if isinstance(entry, dict) and "definition" in entry:
            definition = entry["definition"].strip()
            if min_definition_length <= len(definition) <= max_definition_length:
                results.append(
                    {
                        "word": word,
                        "definition": definition,
                        "part_of_speech": entry.get("part_of_speech", ""),
                        "synonyms": entry.get("synonyms", []),
                    }
                )
    return results


def parse_free_dictionary_entry(
    word, data, min_definition_length=10, max_definition_length=1000
):
    entries = []
    if isinstance(data, list):
        for entry in data:
            if "meanings" in entry:
                for meaning in entry["meanings"]:
                    if "definitions" in meaning:
                        for definition_item in meaning["definitions"]:
                            if "definition" in definition_item:
                                definition = definition_item["definition"].strip()
                                if (
                                    min_definition_length
                                    <= len(definition)
                                    <= max_definition_length
                                ):
                                    entries.append(
                                        {
                                            "word": word,
                                            "definition": definition,
                                            "part_of_speech": meaning.get(
                                                "partOfSpeech", ""
                                            ),
                                            "synonyms": definition_item.get(
                                                "synonyms", []
                                            ),
                                            "antonyms": definition_item.get(
                                                "antonyms", []
                                            ),
                                        }
                                    )
    return entries


def parse_wordsapi_entry(
    word, data, min_definition_length=50, max_definition_length=200
):
    entries = []
    if isinstance(data, dict) and "definitions" in data:
        for definition_item in data["definitions"]:
            if "definition" in definition_item:
                definition = definition_item["definition"].strip()
                if min_definition_length <= len(definition) <= max_definition_length:
                    entries.append(
                        {
                            "word": word,
                            "definition": definition,
                            "part_of_speech": definition_item.get("partOfSpeech", ""),
                            "synonyms": [],  # WordsAPI format may differ
                            "antonyms": [],
                        }
                    )
    return entries


def process_dictionary_data(
    input_file: str = DEFAULT_OUTPUT_FILE,
    output_file: str = DEFAULT_PROCESSED_FILE,
    min_definition_length: int = 50,
    max_definition_length: int = 1000,
    api: str = "free_dictionary",
) -> None:
    """
    Process raw dictionary data into a format suitable for LLM evaluation.

    Args:
        input_file: Path to the raw dictionary data file
        output_file: Path to save the processed data
        min_definition_length: Minimum length of definitions to include
        max_definition_length: Maximum length of definitions to include
        api: API to use for fetching definitions
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return

    with open(input_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_data = []
    for word, data in raw_data.items():
        # Skip entries with errors
        if isinstance(data, dict) and "error" in data:
            continue

        if api == "medical":
            processed_data.extend(
                parse_medical_entry(
                    word, data, min_definition_length, max_definition_length
                )
            )
        elif api == "free_dictionary":
            processed_data.extend(
                parse_free_dictionary_entry(
                    word, data, min_definition_length, max_definition_length
                )
            )
        else:
            processed_data.extend(
                parse_wordsapi_entry(
                    word, data, min_definition_length, max_definition_length
                )
            )

    # Convert to DataFrame and save
    df = pd.DataFrame(processed_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Processed {len(df)} definitions and saved to {output_file}")
    print(f"Dataset contains definitions for {df['word'].nunique()} unique words")


def main():
    """Main function to run the dictionary data collection and processing."""
    parser = argparse.ArgumentParser(description="Collect and process dictionary data")
    parser.add_argument(
        "--word-list", type=str, help="Path to a file containing words (one per line)"
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=2000,
        help="Number of words to fetch if downloading",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["free_dictionary", "words_api", "medical"],
        default="medical",
        help="API to use for fetching definitions",
    )
    parser.add_argument(
        "--max-words", type=int, default=1000, help="Maximum number of words to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between API requests"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection and only process existing data",
    )
    parser.add_argument(
        "--min-definition-length",
        type=int,
        default=50,
        help="Minimum length of definitions to include",
    )
    parser.add_argument(
        "--max-definition-length",
        type=int,
        default=1000,
        help="Maximum length of definitions to include",
    )

    args = parser.parse_args()

    # Update output paths based on output directory
    output_file = os.path.join(args.output_dir, "dictionary_data.json")
    processed_file = os.path.join(args.output_dir, "processed_dictionary.csv")

    if not args.skip_collection:
        # Get word list
        if args.api == "medical":
            word_list = pd.read_csv(
                DEFAULT_MEDICAL_WORD_LIST_FILE, header=None, sep="\t"
            )
            word_list.columns = ["word"]
            word_list = word_list.drop_duplicates().dropna()
            word_list = word_list[word_list["word"].str.isalpha()]
            word_list = word_list["word"].tolist()
        else:
            word_list = get_word_list(args.word_list, args.num_words)

        # Collect dictionary data
        collect_dictionary_data(
            word_list=word_list,
            output_file=output_file,
            api=args.api,
            max_words=args.max_words,
            delay=args.delay,
        )

    # Process dictionary data
    process_dictionary_data(
        input_file=output_file,
        output_file=processed_file,
        min_definition_length=args.min_definition_length,
        max_definition_length=args.max_definition_length,
        api=args.api,
    )


if __name__ == "__main__":
    main()
    # get_wikipedia_medical_glossary_word_list()
