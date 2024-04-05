import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK resources
nltk.download('punkt')

uploaded = ['data.txt']

def preprocess_text(text):
    text = str(text).lower()
    return text

def break_into_chunks(sentences):
    chunks = []
    for i in range(len(sentences)):
        chunk_start = max(0, i - 2)
        chunk_end = min(len(sentences), i + 3)
        base_sentence = sentences[i]
        similar_sentences = find_similar_sentences(base_sentence, sentences[chunk_start:chunk_end])
        chunks.extend(similar_sentences)
    return chunks

def find_similar_sentences(base_sentence, candidate_sentences):
    query_vector = tfidf_vectorizer.transform([base_sentence])
    distances = []
    for sentence in candidate_sentences:
        sentence_vector = tfidf_vectorizer.transform([sentence])
        distance = cosine_similarity(query_vector, sentence_vector)[0][0]
        distances.append((distance, sentence))
    distances.sort(key=lambda x: x[0], reverse=True)
    similar_sentences = [sentence for _, sentence in distances[:2]]  # Return top 2 similar sentences
    return similar_sentences

sentences = []
for filename in uploaded:
    print(f"Uploaded file: {filename}")
    with open(filename, 'r', errors='ignore') as file:
        raw_text = file.read()
        sentences += sent_tokenize(raw_text)
sentences = [preprocess_text(sentence) for sentence in sentences]

# Create TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

# Break sentences into chunks
chunks = break_into_chunks(sentences)
print(len(chunks))
print(len(sentences))

# Calculate centroid vectors for each chunk
chunk_centroids = []
for chunk in chunks:
    chunk_vector = tfidf_vectorizer.transform([chunk])
    chunk_centroid = np.asarray(chunk_vector.mean(axis=0)).reshape(-1)
    chunk_centroids.append(chunk_centroid)

def find_similar_sentences(query_text):
    query_vector = tfidf_vectorizer.transform([query_text])
    distances = []
    for centroid_index, centroid in enumerate(chunk_centroids):
        distance = cosine_similarity(query_vector, centroid.reshape(1, -1))[0][0]
        distances.append((distance, centroid_index))
    distances.sort(key=lambda x: x[0], reverse=True)
    top_chunk_indices = [int(index) for index, _ in distances[:3]]  # Convert indices to integers
    similar_sentences = [chunks[index] for index in top_chunk_indices]
    return similar_sentences


def response(query_text):
    query_text = preprocess_text(query_text)
    similar_sentences = find_similar_sentences(query_text)
    return similar_sentences

query_text = ""
while query_text.lower() != "bye":
    query_text = input("You: ")
    similar_sentences = response(query_text)
    print("Bot: Top similar sentences found:")
    for sentence in similar_sentences:
        print("-", sentence)
