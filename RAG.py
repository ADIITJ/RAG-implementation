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
    text = text.lower()
    return text

def break_into_chunks(sentences):
    chunk_size = 5
    chunks = []
    for i in range(len(sentences)):
        chunk_start = max(0, i - 2)
        chunk_end = min(len(sentences), i + 3)
        chunk = ' '.join(sentences[chunk_start:chunk_end])
        chunks.append(chunk)
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

TfidfVec = TfidfVectorizer(stop_words='english')
tfidf_matrix = TfidfVec.fit_transform(chunks)

# Cluster sentences
num_clusters = len(sentences) // 5  # Adjust as needed
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

def response(query_text):
    query_vector = TfidfVec.transform([query_text])
    
    distances = []
    for cluster_id, cluster in clusters.items():
        distance = cosine_similarity(query_vector, cluster["vector"])[0][0]
        distances.append((distance, cluster_id))
    distances.sort(key=lambda x: x[0], reverse=True)

    closest_cluster = distances[0][1]
    sentences_in_cluster = clusters[closest_cluster]["sentences"]
    robo_response = f"Based on your query, I found these sentences in the closest cluster: \n"
    for s in sentences_in_cluster:
        robo_response += f"- {s}\n"
    return robo_response

query_text = ""
while query_text.lower() != "bye":
    query_text = input("You: ")
    print("Bot:", response(query_text))
