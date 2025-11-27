


"""
Topic modeling on preprocessed Reddit menopause data.
Uses your provided LDA utilities (bigrams, coherence search, pyLDAvis).
Outputs:
  - outputs/lda_final.html  (interactive topic viz)
  - data/processed/menopause_with_topics.csv (adds topic_id & topic_strength)
"""

import os
import re
import pandas as pd
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# ensure NLTK data
nltk.download('stopwords', quiet=True)

# -------------------------
# Config (edit if needed)
# -------------------------
INPUT_CSV = "data/processed/processed_reddit_data.csv"
OUTPUT_CSV = "data/processed/processed_topic_modeling.csv"
OUT_VIS = "outputs/lda_final.html"

# If you want automatic k selection set K_AUTOMATIC=True
K_AUTOMATIC = True
K_FALLBACK = 7     # used only if you set K_AUTOMATIC = False
K_MIN = 3
K_MAX = 10

WORKERS = 10
PASSES = 12
NO_BELOW = 5
NO_ABOVE = 0.5

# ------------------------------------------
# Stopwords (your custom set from snippet)
# ------------------------------------------
stop_words = set(stopwords.words('english')).union({
    'feel', 'go', 'year', 'day', 'start', 'time', 'make', 'get', 'thing',
    'help', 'thank', 'know', 'use', 'need', 'like', 'good', 'bad',
    'menopause', 'perimenopause', 'women', 'woman', 'one', 'also', 'really',
    'still', 'even', 'see', 'say', 'much', 'many', 'well', 'lot'
})

# ------------------------------------------
# Text preprocessing for LDA (from your snippet)
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
def find_best_k(corpus, id2word, texts, start=3, end=10, workers=2):
    print("\n🔍 Finding optimal number of topics...\n")
    scores = []
    for k in range(start, end + 1):
        lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k,
                                 random_state=42, passes=8, workers=workers)
        cm = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        score = cm.get_coherence()
        scores.append((k, score))
        print(f"Num Topics = {k}, Coherence = {score:.4f}")
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\n✅ Best number of topics: {best_k}")

    ks, coherences = zip(*scores)
    plt.figure(figsize=(6,3))
    plt.plot(ks, coherences, marker='o')
    plt.title("Coherence Score vs Number of Topics")
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Coherence Score (c_v)")
    plt.xticks(ks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_k

# ------------------------------------------
# Train final LDA model and save results
# ------------------------------------------
def train_lda(df, out_vis=OUT_VIS, out_csv=OUTPUT_CSV):
    # Use df['text_lem'] (assumes preprocess produced this)
    texts_raw = df['text_lem'].fillna("").astype(str).tolist()
    cleaned = [clean_text(t) for t in texts_raw]
    tokenized = tokenize(cleaned)
    tokenized = make_bigrams(tokenized, min_count=5, threshold=10)

    # build dictionary + corpus
    id2word = corpora.Dictionary(tokenized)
    id2word.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    corpus = [id2word.doc2bow(text) for text in tokenized]

    if len(corpus) == 0 or len(id2word) == 0:
        raise ValueError("Empty corpus or dictionary. Check your input texts and preprocessing.")

    # determine k
    best_k = None
    if K_AUTOMATIC:
        # ensure sensible start/end relative to number of docs
        max_end = min(K_MAX, max(2, len(corpus)//2))
        start = max(K_MIN, 2)
        best_k = find_best_k(corpus, id2word, tokenized, start=start, end=max_end, workers=WORKERS)
    else:
        best_k = K_FALLBACK

    # final model
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=best_k,
        random_state=42,
        passes=PASSES,
        workers=WORKERS
    )

    # coherence
    cm = CoherenceModel(model=lda_model, texts=tokenized, dictionary=id2word, coherence='c_v')
    coherence = cm.get_coherence()
    print(f"\n📈 Final Coherence Score: {coherence:.4f}\n")

    # topic token proportions (pyLDAvis-like ordering)
    def topic_token_proportions(lda_model, corpus):
        n_topics = lda_model.num_topics
        token_counts = np.zeros(n_topics, dtype=float)
        for doc_bow in corpus:
            doc_len = sum(cnt for _, cnt in doc_bow)
            if doc_len == 0:
                continue
            doc_topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)
            for topic_id, prob in doc_topic_dist:
                token_counts[topic_id] += prob * doc_len
        topic_props = token_counts / token_counts.sum()
        order = topic_props.argsort()[::-1]
        return topic_props, order

    topic_props, order = topic_token_proportions(lda_model, corpus)

    # show ordered topics & keywords
    print("Topic ordering (by token proportion):\n")
    topics = lda_model.show_topics(num_topics=best_k, num_words=12, formatted=False)
    for rank, topic_id in enumerate(order, start=1):
        pct = topic_props[topic_id] * 100
        top_words = lda_model.show_topic(topic_id, topn=10)
        top_words_str = ", ".join([w for w, _ in top_words])
        print(f"Vis Topic {rank}  -> Model Topic {topic_id} : {pct:.2f}% tokens")
        print(f"Top words: {top_words_str}\n")

    # save pyLDAvis visualization
    try:
        vis = gensimvis.prepare(lda_model, corpus, id2word)
        os.makedirs(os.path.dirname(out_vis) or ".", exist_ok=True)
        pyLDAvis.save_html(vis, out_vis)
        print(f"\n✅ LDA visualization saved to {out_vis}")
    except Exception as e:
        print("⚠️ pyLDAvis failed to prepare/save visualization:", e)

    # assign dominant topic & strength (probability)
    dominant_topics = []
    topic_strengths = []
    for doc_bow in corpus:
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        if not doc_topics:
            dominant_topics.append(None)
            topic_strengths.append(0.0)
        else:
            top_topic = max(doc_topics, key=lambda x: x[1])
            dominant_topics.append(int(top_topic[0]))   # model topic id
            topic_strengths.append(float(top_topic[1]))

    df_out = df.copy()
    df_out['topic_id'] = dominant_topics
    df_out['topic_strength'] = topic_strengths

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ Data with topics saved to {out_csv}")

    # Return model artifacts for downstream use
    return lda_model, id2word, corpus, tokenized

# ------------------------------------------
# Main
# ------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"🔹 Loaded {len(df)} rows from {INPUT_CSV}")
    lda_model, id2word, corpus, tokenized = train_lda(df, out_vis=OUT_VIS, out_csv=OUTPUT_CSV)
