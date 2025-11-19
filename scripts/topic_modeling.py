import os
import pandas as pd
import re
import nltk
from tqdm import tqdm
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from nltk.corpus import stopwords

# ------------------------------------------
# Ensure required NLTK data is present
# ------------------------------------------
nltk.download('stopwords', quiet=True)

# ------------------------------------------
# Custom stopwords (domain + generic)
# ------------------------------------------
stop_words = set(stopwords.words('english')).union({
    'feel', 'go', 'year', 'day', 'start', 'time', 'make', 'get', 'thing',
    'help', 'thank', 'know', 'use', 'need', 'like', 'good', 'bad',
    'menopause', 'perimenopause', 'women', 'woman', 'one', 'also', 'really',
    'still', 'even', 'see', 'say', 'much', 'many', 'well', 'lot'
})

# ------------------------------------------
# Text preprocessing for LDA
# ------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(texts):
    """Tokenize + remove stopwords"""
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]

def make_bigrams(texts, min_count=5, threshold=10):
    """Create bigram and trigram models"""
    bigram = Phrases(texts, min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[texts], threshold=threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    texts = [bigram_mod[trigram_mod[doc]] for doc in texts]
    return texts

# ------------------------------------------
# Find best number of topics
# ------------------------------------------
def find_best_k(corpus, id2word, texts, start=3, end=10):
    print("\n🔍 Finding optimal number of topics...\n")
    scores = []
    for k in range(start, end + 1):
        lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k,
                                 random_state=42, passes=10, workers=2)
        cm = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        score = cm.get_coherence()
        scores.append((k, score))
        print(f"Num Topics = {k}, Coherence = {score:.3f}")
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\n✅ Best number of topics: {best_k}")


    from matplotlib import pyplot as plt

    ks, coherences = zip(*scores)
    plt.plot(ks, coherences, marker='o')

    plt.title("Coherence Score vs Number of Topics")
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Coherence Score (c_v)")
    plt.xticks(ks)
    plt.grid()
    plt.show()

    return best_k

# ------------------------------------------
# Train final LDA model
# ------------------------------------------
def train_lda(df, out_vis="outputs/lda_final.html", out_csv = "data/processed/menopause_with_topics.csv"):
    texts = [clean_text(t) for t in df['text_lem'].astype(str)]
    tokenized_texts = tokenize(texts)
    tokenized_texts = make_bigrams(tokenized_texts)

    # Dictionary and Corpus
    id2word = corpora.Dictionary(tokenized_texts)
    id2word.filter_extremes(no_below=5, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in tokenized_texts]

    # Tune number of topics
    k = None
    k = 7
    best_k = find_best_k(corpus, id2word, tokenized_texts, start=5, end=10) if k is None else k

    # Final LDA model
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=id2word,
                             num_topics=best_k,
                             random_state=42,
                             passes=15,
                             workers=2)

    # Compute final coherence
    cm = CoherenceModel(model=lda_model, texts=tokenized_texts, dictionary=id2word, coherence='c_v')
    coherence = cm.get_coherence()
    print(f"\n📈 Final Coherence Score: {coherence:.3f}\n")

    # # Display topics
    # for i, topic in :
    #     print(f"Topic {i}: {topic}")

    print("\n🔹 Generating LDA visualization...")

    import numpy as np

    topic_map = {}

    def topic_token_proportions(lda_model, corpus):
        """
        Compute token-weighted topic proportions:
        For each document d, get P(topic|d) from model,
        multiply by doc length (number of tokens in the BoW),
        sum across documents -> token count per topic.
        Normalize to get proportion.
        Returns:
        topic_props: numpy array of shape (num_topics,) with proportions summing to 1
        order: numpy array of topic ids sorted descending by proportion
        """
        n_topics = lda_model.num_topics if hasattr(lda_model, 'num_topics') else lda_model.num_topics
        token_counts = np.zeros(n_topics, dtype=float)
        total_tokens = 0.0

        for doc_bow in corpus:
            doc_len = sum(cnt for _, cnt in doc_bow)   # total tokens in this doc
            if doc_len == 0:
                continue
            total_tokens += doc_len
            # get_document_topics returns list of (topicid, prob). Set minimum_probability=0 to include all topics.
            doc_topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)
            for topic_id, prob in doc_topic_dist:
                token_counts[topic_id] += prob * doc_len

        # Normalize to proportions
        topic_props = token_counts / token_counts.sum()
        order = topic_props.argsort()[::-1]   # topic ids sorted by descending proportion
        return topic_props, order

    # Usage example (after you have lda_model and corpus):
    topic_props, order = topic_token_proportions(lda_model, corpus)


    print("Topic ordering (pyLDAvis-like):\n")
    topics = lda_model.show_topics(num_topics=best_k, num_words=10, formatted=True)
    for rank, topic_id in enumerate(order, start=1):
        pct = topic_props[topic_id] * 100
        top_words = lda_model.show_topic(topic_id, topn=10)
        top_words_str = ", ".join([w for w, _ in top_words])
        print(f"Vis Topic {rank}  -> Model Topic {topic_id} : {pct:.2f}% tokens")
        print(f"Topic Keywords: {topics[topic_id][1]}")
        print(f"Top words: {top_words_str}\n")

        topic_map[topic_id] = rank

    # Save visualization
    vis = gensimvis.prepare(lda_model, corpus, id2word)
    os.makedirs(os.path.dirname(out_vis), exist_ok=True)
    pyLDAvis.save_html(vis, out_vis)
    print(f"\n✅ LDA visualization saved to {out_vis}")

    def get_dominant_topic(lda_model, corpus):
        dominant_topics = []
        topic_percents = []
        for doc_topics in lda_model.get_document_topics(corpus):
            if len(doc_topics) == 0:
                dominant_topics.append(None)
                topic_percents.append(0)
            else:
                top_topic = max(doc_topics, key=lambda x: x[1])
                dominant_topics.append(topic_map[top_topic[0]])
                topic_percents.append(top_topic[1])
        return dominant_topics, topic_percents


    dominant_topics, topic_strength = get_dominant_topic(lda_model, corpus)

    df['topic_id'] = dominant_topics
    df['topic_strength'] = topic_strength
    df.to_csv(out_csv, index=False)
    print(f"\nData with topics saved to {out_csv}")

# ------------------------------------------
# Main
# ------------------------------------------
if __name__ == "__main__":
    input_csv = "data/processed/menopause_processed.csv"
    output_vis = "outputs/lda_final.html"
    output_csv = "data/processed/menopause_with_topics.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"🔹 Loaded {len(df)} posts for topic modeling.")
    train_lda(df, out_vis=output_vis, out_csv=output_csv)