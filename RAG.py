import nltk
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

uploaded = ['data.txt']

sentences = []
word_tokens = []

for filename in uploaded:
    print(f"Uploaded file: {filename}")
    with open(filename, 'r', errors='ignore') as file:
        raw = file.read().lower()
        sentences += sent_tokenize(raw)
        word_tokens += nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()
def LemNormalize(text):
    tokens = nltk.word_tokenize(text.lower().translate(dict((ord(punct), None) for punct in string.punctuation)))
    return [lemmer.lemmatize(token) for token in tokens]

TfidfVec = TfidfVectorizer(stop_words='english')
tfidf_matrix = TfidfVec.fit_transform(sentences)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response = ''
    user_response = user_response.lower()
    sentences.append(user_response)
    tfidf_matrix_new = TfidfVec.fit_transform(sentences)
    vals = cosine_similarity(tfidf_matrix_new[-1], tfidf_matrix_new)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        robo_response += " I am sorry, unable to understand you"
    else:
        closest_cluster = distances[0][1]
        cluster_vector = cluster_vector(closest_cluster)
        similarity = cosine_similarity(tfidf_matrix_new[-1], cluster_vector)[0][0]
        sentences_in_cluster = clusters[closest_cluster]["sentences"]
        robo_response += f"Based on your query, I found these sentences in the closest cluster: \n"
        for s in sentences_in_cluster:
            robo_response += s + "\n"
    sentences.remove(user_response)
    return robo_response

def cluster_vector(cluster_id):
    cluster = clusters[cluster_id]
    vector = cluster["vector"]
    return vector

kmeans = KMeans(n_clusters=5) # Choose the number of clusters that suits your data
kmeans.fit(tfidf_matrix)

cluster_labels = kmeans.labels_

clusters = {}

for i in range(len(cluster_labels)):
    cluster_id = cluster_labels[i]
    if cluster_id not in clusters:
        clusters[cluster_id] = {"sentences": [], "vector": tfidf_matrix[i]}
    else:
        clusters[cluster_id]["sentences"].append(sentences[i])
        clusters[cluster_id]["vector"] = clusters[cluster_id]["vector"] + tfidf_matrix[i]
    clusters[cluster_id]["vector"] = clusters[cluster_id]["vector"] / (len(clusters[cluster_id]["sentences"]) + 1)

distances = []
for cluster_id, cluster in clusters.items():
    distance = cosine_similarity(query_vector, cluster["vector"])[0][0]
    distances.append((distance, cluster_id))
distances.sort(key=lambda x: x[0