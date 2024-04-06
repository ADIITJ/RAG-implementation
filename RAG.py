import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
import re
chunk_centroids = []
chunks = []

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def break_into_chunks(tfidf_vectorizer, sentences):
    chunks = []
    for i in range(len(sentences)):
        chunk_start = max(0, i - 3)
        chunk_end = min(len(sentences), i + 4)
        base_sentence = sentences[i]
        similar_sentences = find_similar_sentence(base_sentence, sentences[chunk_start:chunk_end], tfidf_vectorizer)
        chunks.append(similar_sentences)
        i = i + 4
    return chunks

def find_similar_sentence(base_sentence, candidate_sentences, tfidf_vectorizer):
    query_vector = tfidf_vectorizer.transform([base_sentence])
    distances = []
    for sentence in candidate_sentences:
        sentence_vector = tfidf_vectorizer.transform([sentence])
        distance = cosine_similarity(query_vector, sentence_vector)[0][0]
        distances.append((distance, sentence))
    distances.sort(key=lambda x: x[0], reverse=True)
    similar_sentences = [sentence for _, sentence in distances[:2]]  # Return top 2 similar sentences
    return similar_sentences

def get_file_contents(uploaded):
    global chunks
    global chunk_centroids
    sentences = []
    for filename in uploaded:
        print(f"Uploaded file: {filename}")
        with open(filename, 'r', errors='ignore') as file:
            raw_text = file.read()
            sentences += sent_tokenize(raw_text)
    sentences = [preprocess_text(sentence) for sentence in sentences]
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    chunks = break_into_chunks(tfidf_vectorizer, sentences)
    print(len(chunks))

    for chunk in chunks:
        chunk_vector = tfidf_vectorizer.transform(chunk)
        chunk_centroid = np.asarray(chunk_vector.mean(axis=0)).reshape(-1)
        chunk_centroids.append(chunk_centroid)

def find_similar_sentences(query_text):
    query_vector = tfidf_vectorizer.transform([query_text])
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
    prompt = f"Answer the question: '{query_text}' based on the following context:\n"
    for sentence in similar_sentences:
        prompt += f"{sentence}. "
    return prompt

def response(query_text):
    query_text = preprocess_text(query_text)
    similar_sentences = find_similar_sentences(query_text)
    similar_sentences = convert_data_to_list(similar_sentences)
    rag_prompt = generate_rag_prompt(query_text, similar_sentences)
    return rag_prompt

uploaded = ['data.txt']
get_file_contents(uploaded)

query_text = ""
while True:
    query_text = input("You: ")
    if preprocess_text(query_text)=="bye":
        break
    prompt = response(query_text)
    print("Bot: RAG prompt-")
    print(prompt)
