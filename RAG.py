import sys
import string
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.indexing import VectorStore

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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

def vectorize_text(embedder, text):
    return embedder.encode([text])[0]

def get_similar_sentences(sentence, sentences, embedder, x):
    query_vector = vectorize_text(embedder, sentence)
    distances = []
    for s in sentences:
        vector = vectorize_text(embedder, s)
        distance = cosine_similarity([query_vector], [vector])[0][0]
        distances.append((distance, s))
    distances.sort(key=lambda x: x[0], reverse=True)
    similar_sentences = [s for _, s in distances[:x]]  # Return top 2 similar sentences
    return similar_sentences

def break_into_chunks(embedder, sentences, x):
    chunks = []
    for i in range(0, len(sentences), x):
        chunk_start = max(0, i - x)
        chunk_end = min(len(sentences), i + x+1)
        base_sentence = sentences[i]
        similar_sentences = get_similar_sentences(base_sentence, sentences[chunk_start:chunk_end], embedder, x)
        chunks.append(similar_sentences)
    return chunks

def get_file_contents(uploaded, x):
    sentences = []
    for filename in uploaded:
        print(f"Uploaded file: {filename}")
        with open(filename, 'r', errors='ignore') as file:
            raw_text = file.read()
            lemmatized_text = lemmatize_text(raw_text)
            sentences.extend(sent_tokenize(lemmatized_text))
    
    # Use SentenceTransformer with a pre-trained model (e.g., all-MiniLM-L6-v2)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create a VectorStore
    vectorstore = VectorStore.from_texts(texts=sentences, embedding_model=embedder)
    
    # Use the vector store to compute similarities
    chunks = break_into_chunks(vectorstore, sentences, x)
    return chunks

def find_similar_sentences(query_text, vectorstore, num_chunks):
    query_embedding = vectorstore.embed([query_text])[0]
    distances, indices = vectorstore.search(query_embedding, k=num_chunks)
    similar_sentences = [vectorstore.get_text(index) for index in indices]
    return similar_sentences

def convert_data_to_list(data):
    unique_strings = set()
    for sublist in data:
        for string in sublist:
            unique_strings.add(string)
    return list(unique_strings)

def generate_rag_prompt(query_text, similar_sentences):
    prompt = f"Question: {query_text}\n\n"
    prompt += f"Answers: \n\n"
    for i, sentence in enumerate(similar_sentences):
        prompt += f"{i+1}. {sentence}\n\n"
    return prompt

def process_data(uploaded_files, query_text, num_chunks):
    sentences = get_file_contents(uploaded_files, num_chunks)
    data = convert_data_to_list(sentences)
    sentences = preprocess_text(' '.join(data))
    sentences = sent_tokenize(sentences)
    vectorstore = SentenceTransformer('all-MiniLM-L6-v2')
    similar_sentences = find_similar_sentences(query_text, vectorstore, num_chunks)
    ragged_prompt = generate_rag_prompt(query_text, similar_sentences)
    return ragged_prompt

def process_ragged_prompt(ragged_prompt):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(ragged_prompt)
    print(f"Overall sentiment scores: {scores}")
    print(f"Negative sentiment score: {scores['neg']}")
    print(f"Neutral sentiment score: {scores['neu']}")
    print(f"Positive sentiment score: {scores['pos']}")
    return ragged_prompt

def process_retriever(retriever, prompt):
    result = retriever.predict(prompt)
    print(f"Retriever prediction: {result}")
    return result

def process_final_prompt(retriever, prompt):
    final_prompt = process_ragged_prompt(prompt)
    retriever_prediction = process_retriever(retriever, final_prompt)
    return retriever_prediction

def run_process(retriever, uploaded_files, query_text, num_chunks):
    ragged_prompt = process_data(uploaded_files, query_text, num_chunks)
    result = process_final_prompt(retriever, ragged_prompt)
    return result

if __name__ == "__main__":
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    uploaded_files = sys.argv[1:]
    query_text = "Why are cats sometimes licking themselves?"
    num_chunks = 5
    run_process(retriever, uploaded_files, query_text, num_chunks)