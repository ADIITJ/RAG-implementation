

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

uploaded = files.upload()

sentences = []
word_tokens = []

for filename in uploaded.keys():
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
        robo_response += sentences[idx]
    sentences.remove(user_response)
    return robo_response

# Perform K-means clustering on the TF-IDF vectors
kmeans = KMeans(n_clusters=5) # Choose the number of clusters that suits your data
kmeans.fit(tfidf_matrix)

# Get the cluster labels for each sentence
cluster_labels = kmeans.labels_

# Create a dictionary to store the sentences and their cluster vectors
clusters = {}

# Add each sentence to its corresponding cluster and calculate the cluster vector
for i in range(len(cluster_labels)):
    cluster_id = cluster_labels[i]
    if cluster_id not in clusters:
        clusters[cluster_id] = {"sentences": [], "vector": tfidf_matrix[i]}
    else:
        clusters[cluster_id]["sentences"].append(sentences[i])
        clusters[cluster_id]["vector"] = clusters[cluster_id]["vector"] + tfidf_matrix[i]
    clusters[cluster_id]["vector"] = clusters[cluster_id]["vector"] / (len(clusters[cluster_id]["sentences"]) + 1)

# Now you can input a query vector and find the closest cluster vector
query_vector = TfidfVec.transform(["your query here"])
distances = []
for cluster_id, cluster in clusters.items():
    distance = cosine_similarity(query_vector, cluster["vector"])[0][0]
    distances.append((distance, cluster_id))
distances.sort(key=lambda x: x[0])
closest_cluster = distances[0][1]

# Insert the while loop for user input here
flag = True
print("Bot : Hello, this is a very basic retrieval Chatbot. Enter 'Hi' to start and 'Bye' to end")
while(flag):
    user = input("User : ")
    if(user.lower() == "bye"):
        print("Bot : nice talking to you. tata!!")
        flag = False
    else:
        if(user.lower() == "thank you" or user.lower() == "thanks"):
            print("Bot : You are welcome")
        else:
            if(user.lower() == "hi"):
                print("Bot : Hello! How can I help you?")
            else:
                print("Bot : ", end="")
                print(response(user))







