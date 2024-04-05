## RAG Implementation

This repository contains an implementation of the Retrieval-Augmented Generation (RAG) model, which is used for question answering and generation tasks. The RAG model combines the power of retrieval-based and generation-based approaches to provide accurate and contextually relevant responses.

### Requirements
- Python 3.x
- NLTK
- scikit-learn
- NumPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ADIITJ/RAG-implementation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd RAG-implementation
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK resources:
   ```bash
   python -m nltk.downloader punkt
   ```

### Usage

1. Ensure that your data is stored in a file named `data.txt`.
2. Run the main script:
   ```bash
   python RAG.py
   ```
3. Enter your query when prompted by the program.
4. The program will display the RAG prompt and the top contextually similar sentences.

### How It Works

- **Preprocessing**: The text data is preprocessed to convert it to lowercase, remove special characters and extra whitespaces.
- **Chunking**: The text is divided into chunks of sentences, with each chunk containing a base sentence and its similar sentences.
- **Similarity Calculation**: The TF-IDF vectorizer is used to calculate the similarity between sentences. Cosine similarity is used as the similarity metric.
- **Centroid Calculation**: Centroid vectors are calculated for each chunk using the TF-IDF matrix.
- **Response Generation**: Given a query, the program finds similar sentences and presents them as a context for answering the query.


### Acknowledgments

- Inspired by the RAG model proposed by Lewis et al. (2020).
- Thanks to the NLTK and scikit-learn communities for their invaluable contributions to natural language processing and machine learning.
