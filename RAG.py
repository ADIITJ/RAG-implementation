import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import string
import re
from langchain.llm import LLM
from langchain.embeddings import HuggingFaceEmbeddings
import cosine_similarity
nltk.download('punkt')
nltk.download('wordnet')

chunk_centroids = []
chunks = []
embeddings = HuggingFaceEmbeddings("bert-base-uncased")
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def LemNormalize(text):
    llm = LLM("llm-large")
    tokens = word_tokenize(text.lower().translate(dict((ord(punct), None) for punct in string.punctuation)))
    return [llm.lemmatize(token) for token in tokens]

def break_into_chunks(embeddings, sentences, x):
    chunks = []
    for i in range(len(sentences)):
        chunk_start = max(0, i - x)
        chunk_end = min(len(sentences), i + x+1)
        base_sentence = sentences[i]
        similar_sentences = find_similar_sentence(base_sentence, sentences[chunk_start:chunk_end], embeddings, x)
        chunks.append(similar_sentences)
        i = i + x//2
    return chunks

def find_similar_sentence(base_sentence, candidate_sentences, embeddings, x):
    query_vector = embeddings.encode([base_sentence])
    distances = []
    for sentence in candidate_sentences:
        sentence_vector = embeddings.encode([sentence])
        distance = cosine_similarity(query_vector, sentence_vector)[0][0]
        distances.append((distance, sentence))
    distances.sort(key=lambda x: x[0], reverse=True)
    similar_sentences = [sentence for _, sentence in distances[:x]]  # Return top 2 similar sentences
    return similar_sentences

def get_file_contents(uploaded, x):
    global chunks
    global chunk_centroids
    sentences = []
    for filename in uploaded:
        print(f"Uploaded file: {filename}")
        with open(filename, 'r', errors='ignore') as file:
            raw_text = file.read()
            raw_sentences = sent_tokenize(raw_text)
            lemmatized_sentences = []
            for raw_sentence in raw_sentences:
                lemmatized_sentence = ' '.join(LemNormalize(raw_sentence))
                lemmatized_sentences.append(lemmatized_sentence)
            sentences += lemmatized_sentences
    chunks = break_into_chunks(embeddings, sentences, x)

    for chunk in chunks:
        chunk_vector = embeddings.encode(chunk)
        chunk_centroid = np.asarray(chunk_vector.mean(axis=0)).reshape(-1)
        chunk_centroids.append(chunk_centroid)


def find_similar_sentences(query_text, num_chunks):
    query_vector = embeddings.encode([query_text])
    distances = []
    for i in range(len(chunk_centroids)):
        distance = cosine_similarity(query_vector, chunk_centroids[i].reshape(1, -1))[0][0]
        distances.append((distance, i))
    distances.sort(key=lambda x: x[0], reverse=True)
    top_chunk_indices = [i for _, i in distances[:num_chunks]]  # Convert indices to integers
    similar_sentences = [chunks[index] for index in top_chunk_indices]
    return similar_sentences

def convert_data_to_list(data):
    unique_strings = set()
    for sublist in data:
        for string in sublist:
            unique_strings.add(string)
    return list(unique_strings)

def generate_rag_prompt(query_text, similar_sentences):
    prompt = f"The given query is: '{query_text}'\n"
    prompt += f"Here are {len(similar_sentences)} sentences similar to the query: \n"
    for i, sentence in enumerate(similar_sentences):
        prompt += f"{i+1}. {sentence}\n"
    prompt += "\n\nWrite a summarizing paragraph using the above sentences. (Limit: 200 words)\n\n"
    return prompt

def summarize_sentences(prompt, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main(uploaded_files, sentences_per_chunk, num_chunks):
    x = sentences_per_chunk  # Number of sentences to consider in a chunk

    get_file_contents(uploaded_files, x)

    query_text = input("Enter your query: ")
    similar_sentences = find_similar_sentences(query_text, num_chunks)

    data = convert_data_to_list(similar_sentences)
    prompt = generate_rag_prompt(query_text, data)
    summary = summarize_sentences(prompt)

    print("\n\nGenerated Summary:")
    print(summary)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python program.py <uploaded_files> <sentences_per_chunk> <num_chunks>")
        sys.exit(1)
    
    uploaded_files = sys.argv[1].split(',')
    sentences_per_chunk = int(sys.argv[2])
    num_chunks = int(sys.argv[3])
    main(uploaded_files, sentences_per_chunk, num_chunks)

