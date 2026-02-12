# Cluster publications from each region based on their content summaries (descriptions)
# Following tutorial Understanding, Generating, and Visualizing Embeddings by Mike Levy (https://www.dataquest.io/blog/understanding-generating-and-visualizing-embeddings/)

! pip install sentence-transformers scikit-learn matplotlib numpy

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
df = pd.read_csv("publications-asia.csv")

titles = df.iloc[:, 0].astype(str).values
descriptions = df.iloc[:, 1].astype(str).values

# Generate embeddings for content summaries
embeddings = model.encode(descriptions)

# Clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=1)
cluster_labels = kmeans.fit_predict(embeddings)

# Dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(20, 16))

scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=cluster_labels,
    cmap='Set3',
    s=100,
    alpha=0.90,
)

df["cluster"] = cluster_labels

cluster_counts = df["cluster"].value_counts().sort_index()
print(cluster_counts)


# Publication titles
for i, (x, y) in enumerate(embeddings_2d):
    plt.text(
        x,
        y,
        titles[i],
        verticalalignment='center',
        fontsize=10,
        alpha=0.4
    )

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
