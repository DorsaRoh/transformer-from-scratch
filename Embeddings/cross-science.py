import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import re

# Load data
data_dir = 'data/arxiv_abstracts/'
parquet_files = glob(os.path.join(data_dir, '*.parquet'))

dfs = []  
for file in parquet_files:
    df = pd.read_parquet(file)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

print(f"Total number of papers: {len(full_df)}")


embeddings_array = np.vstack(full_df['embeddings'].values) # Prepare embeddings

print("Shape of embeddings array:", embeddings_array.shape)

# Reduce dimensionality for visualization (2D, from 768-dimensions)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Visualize embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.1)
plt.title("2D Visualization of arXiv Paper Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# Cluster embeddings
n_clusters = 20  # Number of scientific fields and/or topics of study
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_array)
full_df['cluster'] = cluster_labels

# Visualize clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, alpha=0.1, cmap='viridis')
plt.colorbar(scatter)
plt.title("Clustered 2D Visualization of arXiv Paper Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# Analyze clusters
def get_common_words(abstracts, n=10):
    words = ' '.join(abstracts).lower()
    words = re.findall(r'\w+', words)
    return Counter(words).most_common(n)

for cluster in range(n_clusters):
    cluster_abstracts = full_df[full_df['cluster'] == cluster]['abstract']
    common_words = get_common_words(cluster_abstracts)
    print(f"Cluster {cluster} common words: {common_words}")

# Find cross-cluster neighbors
def find_cross_cluster_neighbors(embeddings, clusters, n_neighbors=5):
    similarities = cosine_similarity(embeddings)
    cross_cluster_neighbors = []
    
    for i in range(len(embeddings)):
        current_cluster = clusters[i]
        neighbor_similarities = [(j, similarities[i][j]) for j in range(len(embeddings)) if clusters[j] != current_cluster]
        top_neighbors = sorted(neighbor_similarities, key=lambda x: x[1], reverse=True)[:n_neighbors]
        cross_cluster_neighbors.append(top_neighbors)
    
    return cross_cluster_neighbors

cross_cluster_neighbors = find_cross_cluster_neighbors(embeddings_array, cluster_labels)

# Add cross-cluster neighbors to the dataframe
full_df['cross_cluster_neighbors'] = cross_cluster_neighbors

# Create cluster connection graph
def create_cluster_graph(cross_cluster_neighbors, clusters, n_clusters):
    G = nx.Graph()
    for i in range(n_clusters):
        G.add_node(i)
    
    for i, neighbors in enumerate(cross_cluster_neighbors):
        source_cluster = clusters[i]
        for j, similarity in neighbors:
            target_cluster = clusters[j]
            if G.has_edge(source_cluster, target_cluster):
                G[source_cluster][target_cluster]['weight'] += similarity
            else:
                G.add_edge(source_cluster, target_cluster, weight=similarity)
    
    return G

cluster_graph = create_cluster_graph(cross_cluster_neighbors, cluster_labels, n_clusters)

# Visualize cluster graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(cluster_graph)
nx.draw(cluster_graph, pos, with_labels=True, node_color='lightblue', 
        node_size=1000, font_size=12, font_weight='bold')
edge_weights = nx.get_edge_attributes(cluster_graph, 'weight')
nx.draw_networkx_edge_labels(cluster_graph, pos, edge_labels=edge_weights)
plt.title("Connections between Clusters (Scientific Fields)")
plt.show()

# Find unexpected connections
def find_unexpected_connections(cross_cluster_neighbors, clusters, full_df, top_n=10):
    connections = []
    for i, neighbors in enumerate(cross_cluster_neighbors):
        for j, similarity in neighbors:
            connections.append((i, j, similarity, abs(clusters[i] - clusters[j])))
    
    # Sort by similarity (descending) and cluster difference (descending)
    connections.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    print("Top unexpected connections:")
    for i, j, similarity, cluster_diff in connections[:top_n]:
        print(f"Paper 1 (Cluster {clusters[i]}): {full_df.iloc[i]['abstract'][:100]}...")
        print(f"Paper 2 (Cluster {clusters[j]}): {full_df.iloc[j]['abstract'][:100]}...")
        print(f"Similarity: {similarity:.4f}, Cluster difference: {cluster_diff}")
        print()

find_unexpected_connections(cross_cluster_neighbors, cluster_labels, full_df)