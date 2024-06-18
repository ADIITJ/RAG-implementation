# README

## Overview

This project processes text files to generate contextually relevant prompts using hierarchical clustering and machine learning techniques. It analyzes text data, breaks it into chunks, finds similar sentences, and generates a prompt to answer specific questions based on the provided context. Additionally, it performs sentiment analysis on the generated prompt.

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - nltk
  - scikit-learn
  - sentence-transformers

## Setup

1. **Install Python packages**:
   Run the following command to install the required packages:
   ```sh
   pip install numpy nltk scikit-learn sentence-transformers
   ```

2. **Download NLTK data**:
   The script automatically downloads the necessary NLTK data. Make sure your internet connection is active when you run the script for the first time.

## Script Description

### Main Script Components

- **Imports**:
  The script starts by importing necessary libraries including `nltk`, `numpy`, `re`, `string`, and `sentence-transformers`.

- **NLTK Downloads**:
  The script downloads required NLTK data for tokenization, lemmatization, stopwords, and sentiment analysis.

- **Sentence Transformer Model**:
  A pre-trained model (`all-MiniLM-L6-v2`) from `sentence-transformers` is loaded to generate sentence embeddings.

- **Functions**:
  - `lemmatize_text(text)`: Lemmatizes and removes stopwords from the text.
  - `preprocess_text(text)`: Preprocesses the text by converting it to lowercase and removing punctuation.
  - `vectorize_text(text)`: Converts text to embeddings using the SentenceTransformer model.
  - `get_similar_sentences(sentence, sentences, num_sentences)`: Finds similar sentences to the input sentence from a list of sentences.
  - `break_into_chunks(sentences, x)`: Breaks text into chunks based on sentence similarity.
  - `get_file_contents(uploaded, x)`: Reads and processes text from uploaded files, breaking it into chunks.
  - `find_similar_sentences(query_text, embeddings)`: Finds sentences similar to the query text using chunk centroids.
  - `convert_data_to_list(data)`: Converts nested lists of strings into a unique list.
  - `generate_rag_prompt(query_text, similar_sentences)`: Generates a context-based prompt from similar sentences.
  - `process_data(uploaded_files, x)`: Processes uploaded files and generates the initial prompt.
  - `process_ragged_prompt(ragged_prompt)`: Analyzes sentiment of the generated prompt.
  - `process_retriever(prompt)`: Retrieves predictions based on the prompt.
  - `process_final_prompt(prompt)`: Combines prompt processing and retriever prediction.
  - `run_process(uploaded_files, x)`: Main function to run the entire process.

### Usage

1. **Run the script**:
   ```sh
   python script.py file1.txt file2.txt ...
   ```

2. **Provide Input**:
   - When prompted, enter the number of sentences per chunk.
   - Enter your query question when prompted.

### Example

To run the script with example files:
```sh
python script.py example1.txt example2.txt
```

### Output

- The script will print the context-based prompt, sentiment scores, and retriever predictions.

### Notes

- Make sure the input text files are in plain text format.
- The script handles errors in reading files by using `errors='ignore'` in the file open method.