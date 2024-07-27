#basic imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
sns.set()
import os
import sys
import tomotopy as tp
import os
import re
import pyLDAvis
import seaborn as sns

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    return normalized_scores

def calculate_relevance(topic_term_dists, term_freqs):
    lambdas = [0.,0.25, 0.5, 0.75]
    results = []
    """Calculate relevance for each term in each topic."""
    for lams in lambdas: 
        topic_term_dists = np.array(topic_term_dists)
        term_freqs = np.array(term_freqs)
        term_freqs = term_freqs / term_freqs.sum()  # Normalize term frequencies
        relevance = lams * np.log(topic_term_dists) + (1 - lams) * np.log(topic_term_dists / term_freqs)
        normalized_relevance = normalize_scores(relevance)
        results.append(normalized_relevance)
    return results 

def calculate_saliency(topic_term_dists, term_freqs):
    term_freqs = np.array(term_freqs)
    term_freqs = term_freqs / term_freqs.sum()  # Normalize term frequencies

    num_topics, num_terms = topic_term_dists.shape
    saliency = np.zeros_like(topic_term_dists)
    for k in range(num_topics):  # For each topic
        for w in range(num_terms):  # For each word in the vocabulary
            p_word_given_topic = topic_term_dists[k, w]
            p_word_in_corpus = term_freqs[w]
            saliency[k, w] += p_word_given_topic * np.log(p_word_given_topic / p_word_in_corpus)

    normalized_saliency = normalize_scores(saliency)
    return normalized_saliency

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    stop_words = set(["a", "an", "the", "and", "or", "but", "if", "on", "in", "to", "is", "of", "for"])
    words = [word for word in text.split() if len(word) > 2 and word not in stop_words]
    return words


def lda_example(file_paths, timestamps):
    mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=15, k=15)#,#tw=tp.TermWeight.IDF)#alpha=0.1, eta=0.01)
    doc_timestamps = []
    for file_path, timestamp in zip(file_paths, timestamps):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                words = preprocess_text(line.strip())
                if words:
                    mdl.add_doc(words)
                    doc_timestamps.append(timestamp)
    mdl.burn_in = 500
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    mdl.train(100, show_progress=True)
    mdl.summary()

    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq

    
    relevance = calculate_relevance(topic_term_dists, term_frequency)
    saliency = calculate_saliency(topic_term_dists, term_frequency)
    
    print(relevance, saliency)
    for k in range(mdl.k):
        print(f'Topic #{k}')
        words_probs = mdl.get_topic_words(k)
        top_prob_words = [(word, prob) for word, prob in words_probs[:10]]

        relevance_scores = relevance[k]
        relevance_indices = np.argsort(relevance_scores)[::-1]
        saliency_scores = saliency[k]
        saliency_indices = np.argsort(saliency_scores)[::-1]

        top_relevance_words = [(vocab[idx], relevance_scores[idx]) for idx in relevance_indices[:10]]
        top_saliency_words = [(vocab[idx], saliency_scores[idx]) for idx in saliency_indices[:10]]

        prob_words = [word for word, _ in top_prob_words]
        prob_values = [score for _, score in top_prob_words]
        relevance_words = [word for word, _ in top_relevance_words]
        relevance_values = [score for _, score in top_relevance_words]
        saliency_words = [word for word, _ in top_saliency_words]
        saliency_values = [score for _, score in top_saliency_words]

        # Create DataFrame with words and scores for all three metrics
        combined_df = pd.DataFrame({
            'Probability Words': prob_words,
            'Relevance Words': relevance_words,
            'Saliency Words': saliency_words,
            'Probability Scores': prob_values,
            'Relevance Scores': relevance_values,
            'Saliency Scores': saliency_values
        })
        print(combined_df.to_string(index=False))
    prepared_data = pyLDAvis.prepare(
        topic_term_dists, 
        doc_topic_dists, 
        doc_lengths, 
        vocab, 
        term_frequency,
        start_index=0, 
        sort_topics=False 
    )
    pyLDAvis.save_html(prepared_data, 'ldavis.html')
    # Aggregate topic distributions by time periods
    df = pd.DataFrame(doc_topic_dists, columns=[f'Topic_{i}' for i in range(mdl.k)])
    df['Timestamp'] = doc_timestamps
    
    # Calculate the average topic distribution for each time period
    topic_over_time = df.groupby('Timestamp').mean().reset_index()
    
    # Smooth out the frequency lines using a rolling average
    topic_over_time_smooth = topic_over_time.set_index('Timestamp').rolling(window=6, min_periods=1).mean().reset_index()
    
    # Plot the topic frequencies over time
    plt.figure(figsize=(12, 8))
    for topic in topic_over_time_smooth.columns[1:]:
        plt.plot(topic_over_time_smooth['Timestamp'], topic_over_time_smooth[topic], label=topic)
    plt.xlabel('Time Period')
    plt.ylabel('Average Topic Proportion')
    plt.title('Topic Frequencies Over Time (Smoothed)')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(14, 10))  # Adjust figure size for better visibility
    df_melted = topic_over_time_smooth.melt(id_vars='Timestamp', var_name='Topic', value_name='Proportion')
    g = sns.FacetGrid(df_melted, col='Topic', col_wrap=4, sharey=False, height=3)
    g.map(sns.lineplot, 'Timestamp', 'Proportion')
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels('Time Period', 'Average Topic Proportion')
    g.fig.suptitle('Topic Frequencies Over Time (Smoothed)', y=1.02)  # Title with better positioning
    g.add_legend()
    plt.tight_layout()
    plt.show()


def get_all_files_in_directory(directory):
    file_paths = []
    timestamps = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
                timestamp = os.path.basename(root)
                timestamps.append(timestamp)
    return file_paths, timestamps

# Path to the main directory containing subdirectories with text files
main_directory_path = './PSB_Papers/main_body'

print('Running LDA')

# Get all file paths in the main directory and its subdirectories
file_paths, timestamps = get_all_files_in_directory(main_directory_path)

# Run the LDA model on all files
lda_example(file_paths, timestamps)