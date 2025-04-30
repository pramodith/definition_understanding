# Definition Understanding

A project to evaluate how well Large Language Models (LLMs) understand word definitions.

## Overview

This project tests LLMs' ability to predict words based on their definitions. The methodology involves:

1. Curating a dataset of words and their definitions from dictionary sources
2. Presenting definitions to LLMs and asking them to predict the corresponding words
3. Evaluating performance using various metrics
4. Accounting for synonyms in the evaluation process

## Project Structure

- `data/`: Contains raw and processed dictionary datasets
- `src/`: Source code for the project
  - `data_processing/`: Scripts for data acquisition and preprocessing
  - `evaluation/`: Modules for evaluating LLM performance
  - `models/`: Interfaces to different LLMs
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `tests/`: Unit tests

## Getting Started

1. Set up the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

2. Run the data collection script:
   ```bash
   python src/data_processing/collect_dictionary.py
   ```

3. Evaluate an LLM:
   ```bash
   python src/evaluation/evaluate_model.py --model [MODEL_NAME]
   ```

## License

MIT