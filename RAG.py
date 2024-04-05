import nltk
import string
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

uploaded = ['data.txt']

def preprocess_text(text):
    text = str(text).lower()
    return text

def break_into_chunks(sentences):
    chunk_size = 5
    chunks = []
    for i in range(len(sentences)):
        chunk_start = max(0, i - 2)
        chunk_end = min(len(sentences), i + 3)
        base_sentence = sentences[i]
        similar_sentences = find_similar_sentences(base_sentence, sentences[chunk_start:chunk_end])
        chunks.extend(similar_sentences)
    return chunks

    
sentences = []
for filename in uploaded:
    print(f"Uploaded file: {filename}")
    with open(filename, 'r', errors='ignore') as file:
        raw_text = file.read()
        sentences += sent_tokenize(raw_text)
sentences = [preprocess_text(sentence) for sentence in sentences]

# Break sentences into chunks
chunks = break_into_chunks(sentences)
print(len(chunks))
print(len(sentences))
TfidfVec = TfidfVectorizer(stop_words='english')
tfidf_matrix = TfidfVec.fit_transform(chunks)

num_clusters = len(chunks)  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)
cluster_labels = kmeans.labels_

clusters = {}


for i in range(len(cluster_labels)):
    cluster_id = cluster_labels[i]
    if cluster_id not in clusters:
        clusters[cluster_id] = {"sentences": [sentences[i]], "vector": tfidf_matrix[i]}
    else:
        clusters[cluster_id]["sentences"].append(sentences[i])
        clusters[cluster_id]["vector"] += tfidf_matrix[i]

# Calculate centroid vectors for each cluster
for cluster_id, cluster in clusters.items():
    cluster["vector"] /= len(cluster["sentences"])

def find_similar_sentences(query_text, cluster):
    query_vector = TfidfVec.transform([query_text])
    distances = []
    for sentence in cluster["sentences"]:
        sentence_vector = TfidfVec.transform([sentence])
        distance = cosine_similarity(query_vector, sentence_vector)[0][0]
        distances.append((distance, sentence))
    distances.sort(key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in distances[:3]]  # Return top 3 similar sentences
    
def response(query_text):
    query_text = preprocess_text(query_text)
    results = []
    for cluster_id, cluster in clusters.items():
        similar_sentences = find_similar_sentences(query_text, cluster)
        results.extend(similar_sentences)
    return results


query_text = ""
while query_text.lower() != "bye":
    query_text = input("You: ")
    similar_sentences = response(query_text)
    print("Bot: Top similar sentences found:")
    for sentence in similar_sentences:
        print("-", sentence)

