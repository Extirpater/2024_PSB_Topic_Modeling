import os
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter

# Constants
MAIN_BODY_FOLDER = '/Users/DevZhang/data-retrieval-project/PSB_Papers/main_body'
NUM_CLUSTERS = 5  # Number of clusters for K-means
BATCH_SIZE = 10  # Number of text files to process in each batch

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def load_main_body_text(folder_path):
    """
    Load main body text from text files in a folder.
    """
    main_body_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                main_body_texts.append({
                    'filename': filename,
                    'text': f.read()
                })
    return main_body_texts

def get_bert_embeddings(texts):
    """
    Get BERT embeddings for a list of texts.
    """
    tokenized_texts = [tokenizer.encode(text['text'], add_special_tokens=True, max_length=512) for text in texts]
    max_len = max(len(tokens) for tokens in tokenized_texts)
    padded_texts = [tokens + [0]*(max_len - len(tokens)) for tokens in tokenized_texts]
    attention_masks = [[1]*len(tokens) + [0]*(max_len - len(tokens)) for tokens in tokenized_texts]

    input_ids = torch.tensor(padded_texts)
    attention_masks = torch.tensor(attention_masks)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    embeddings = outputs[0][:, 0, :].numpy()  # Extract embeddings for [CLS] tokens

    return embeddings

def bert_topic_modeling(texts, num_clusters=5):
    """
    Perform BERT-based topic modeling using K-means clustering.
    """
    embeddings = get_bert_embeddings(texts)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_

    return embeddings, cluster_labels, cluster_centers

def save_topics(texts, cluster_labels, output_folder):
    """
    Save the topics (cluster assignments) to files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for text, label in zip(texts, cluster_labels):
        output_path = os.path.join(output_folder, f"{text['filename'][:-4]}_topic_{label}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text['text'])

def visualize_clusters(embeddings, cluster_labels, num_clusters):
    """
    Visualize clusters using PCA.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for cluster in range(num_clusters):
        plt.scatter(pca_result[cluster_labels == cluster, 0], pca_result[cluster_labels == cluster, 1],
                    color=colors[cluster], label=f'Cluster {cluster}')

    plt.title('BERT Topic Modeling Clusters (PCA Visualization)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def extract_cluster_keywords(embeddings, cluster_labels, num_clusters, num_keywords=10):
    """
    Extract keywords for each cluster based on the closest words to cluster centroids.
    """
    cluster_keywords = {}

    for cluster in range(num_clusters):
        cluster_embeddings = embeddings[cluster_labels == cluster]
        centroid = np.mean(cluster_embeddings, axis=0)
        closest_embeddings = cluster_embeddings[np.argsort(np.linalg.norm(cluster_embeddings - centroid, axis=1))[:num_keywords]]
        closest_tokens = [tokenizer.convert_ids_to_tokens(tokens) for tokens in closest_embeddings]

        # Flatten and count token occurrences
        token_counter = Counter([token for sublist in closest_tokens for token in sublist])

        # Select the most common tokens as keywords
        top_keywords = [token for token, count in token_counter.most_common(num_keywords)]
        cluster_keywords[cluster] = top_keywords

    return cluster_keywords

def visualize_cluster_keywords(cluster_keywords):
    """
    Visualize cluster keywords using word clouds.
    """
    for cluster, keywords in cluster_keywords.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(' '.join(keywords))

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Cluster {cluster} Keywords')
        plt.axis("off")
        plt.show()

def main():
    # Load main body texts
    main_body_texts = load_main_body_text(MAIN_BODY_FOLDER)

    # Process texts in batches
    for i in range(0, len(main_body_texts), BATCH_SIZE):
        batch_texts = main_body_texts[i:i + BATCH_SIZE]

        # Perform BERT-based topic modeling
        embeddings, cluster_labels, cluster_centers = bert_topic_modeling(batch_texts, NUM_CLUSTERS)

        # Save topics to files
        output_folder = './PSB_Papers/topics'
        save_topics(batch_texts, cluster_labels, output_folder)

        # Visualize clusters using PCA
        visualize_clusters(embeddings, cluster_labels, NUM_CLUSTERS)

        # Extract and visualize cluster keywords
        cluster_keywords = extract_cluster_keywords(embeddings, cluster_labels, NUM_CLUSTERS)
        visualize_cluster_keywords(cluster_keywords)

        # Print cluster labels for each main body text
        for idx, label in enumerate(cluster_labels):
            print(f"Text {i+idx+1}: Cluster {label}")

if __name__ == "__main__":
    main()
