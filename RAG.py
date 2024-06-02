import sys
import string
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
# Use a pipeline as a high-level helper
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_centroids = []
chunks = []
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized_tokens)

def preprocess_text(text):
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text.lower())
    text = lemmatize_text(text)
    return text

def vectorize_text(text):
    return np.expand_dims(embedder.encode(text), axis=0)

def get_similar_sentences(sentence, sentences,num_sentences):
    query_vector = vectorize_text(sentence)
    distances = []
    for s in sentences:
        vector = vectorize_text(s)
        distance = cosine_similarity(query_vector, vector)[0][0]
        distances.append((distance, s))
    distances.sort(key=lambda x: x[0], reverse=True)
    similar_sentences = [s for _, s in distances[:num_sentences]]
    return similar_sentences

def break_into_chunks(sentences, x):
    for i in range(len(sentences)):
        chunk_start = max(0, i - x)
        chunk_end = min(len(sentences), i + x+1)
        base_sentence = sentences[i]
        similar_sentences = get_similar_sentences(base_sentence, sentences[chunk_start:chunk_end], x)
        chunks.append(similar_sentences)
        i = i + x//2 + 1

def get_file_contents(uploaded, x):
    sentences = []
    for filename in uploaded:
        print(f"Uploaded file: {filename}")
        with open(filename, 'r', errors='ignore') as file:
            raw_text = file.read()
            lemmatized_text = lemmatize_text(raw_text)
            sentences.extend(sent_tokenize(lemmatized_text))
    
    break_into_chunks(sentences, x)
    for chunk in chunks:
        # Generate embeddings for each sentence in the chunk
        chunk_embeddings = [vectorize_text(sentence) for sentence in chunk]
        
        # Calculate the centroid of the chunk by averaging the embeddings
        chunk_centroid = np.mean(chunk_embeddings, axis=0)
        
        # Append the centroid to the list of chunk centroids
        chunk_centroids.append(chunk_centroid)

def find_similar_sentences(query_text, embeddings):
    query_vector = vectorize_text(query_text)
    distances = []
    for i in range(len(chunk_centroids)):
        distance = cosine_similarity(query_vector, chunk_centroids[i].reshape(1, -1))[0][0]
        distances.append((distance, i))
    distances.sort(key=lambda x: x[0], reverse=True)
    top_chunk_indices = [i for _, i in distances[:5]]  # Convert indices to integers
    similar_sentences = [chunks[index] for index in top_chunk_indices]
    return similar_sentences

def convert_data_to_list(data):
    unique_strings = set()
    for sublist in data:
        for string in sublist:
            unique_strings.add(string)
    return list(unique_strings)

def generate_rag_prompt(query_text, similar_sentences):
    prompt = f"Context: \n\n"
    for i, sentence in enumerate(similar_sentences):
        prompt += f"{i+1}. {sentence}\n\n"
    prompt += f"Question: {query_text}\nAnswer the question based on the provided context only:"
    print(prompt)
    return prompt

def process_data(uploaded_files, x):
    get_file_contents(uploaded_files, x)
    query = input('Your Question : ')
    similar_sentences = find_similar_sentences(query, embedder)
    final_sentences = convert_data_to_list(similar_sentences)
    ragged_prompt = generate_rag_prompt(query, final_sentences)
    return ragged_prompt

def process_ragged_prompt(ragged_prompt):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(ragged_prompt)
    print(f"Overall sentiment scores: {scores}")
    print(f"Negative sentiment score: {scores['neg']}")
    print(f"Neutral sentiment score: {scores['neu']}")
    print(f"Positive sentiment score: {scores['pos']}")
    return ragged_prompt

def process_retriever(prompt):
    result = embedder.predict(prompt)
    print(f"Retriever prediction: {result}")
    return result

def process_final_prompt(prompt):
    final_prompt = process_ragged_prompt(prompt)
    retriever_prediction = process_retriever(final_prompt)
    return retriever_prediction

def run_process(uploaded_files, x):
    ragged_prompt = process_data(uploaded_files, x)
    result = process_final_prompt(ragged_prompt)
    return result

if __name__ == "__main__":
    uploaded_files = sys.argv[1:]
    num_sentences = int(input("number of sentences per chunk : "))
    run_process(uploaded_files, num_sentences)