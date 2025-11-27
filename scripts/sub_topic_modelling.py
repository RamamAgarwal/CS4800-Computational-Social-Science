"""
Subtopic modeling (comment-level) per main topic.
Input:
  - data/processed/menopause_with_topics.csv (produced by topic_modeling.py)
Output:
  - data/processed/menopause_with_topics_and_subtopics.csv (adds comment_subtopic)
"""

import os
import re
import pandas as pd
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

nltk.download('stopwords', quiet=True)

# -------------------------
# Config (edit if needed)
# -------------------------
INPUT_CSV = "data/processed/processed_topic_modeling.csv"
OUTPUT_CSV = "data/processed/processed_subtopic_modeling.csv"

WORKERS = 10
PASSES = 10
NO_BELOW = 5
NO_ABOVE = 0.5

MIN_DOCS_PER_TOPIC = 30
SUB_MIN_K = 2
SUB_MAX_K = 8

# reuse stopwords (mirror of topic_modeling)
stop_words = set(stopwords.words('english')).union({
    'feel', 'go', 'year', 'day', 'start', 'time', 'make', 'get', 'thing',
    'help', 'thank', 'know', 'use', 'need', 'like', 'good', 'bad',
    'menopause', 'perimenopause', 'women', 'woman', 'one', 'also', 'really',
    'still', 'even', 'see', 'say', 'much', 'many', 'well', 'lot'
})

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize_docs(texts):
    return [
        [w for w in simple_preprocess(str(doc)) if w not in stop_words]
        for doc in texts
    ]

def make_bigrams_tokenized(texts, min_count=5, threshold=10):
    bigram = Phrases(texts, min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[texts], threshold=threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    return [bigram_mod[trigram_mod[doc]] for doc in texts]

def find_best_k_for_texts(texts_tokenized, start=SUB_MIN_K, end=SUB_MAX_K, workers=2):
    dictionary = corpora.Dictionary(texts_tokenized)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    corpus = [dictionary.doc2bow(t) for t in texts_tokenized]
    if len(corpus) == 0 or len(dictionary) == 0 or len(corpus) < 3:
        return None, None, None, 0

    # limit end to sensible bound
    max_end = min(end, max(2, len(corpus)//2))
    best_k = None
    best_score = -999
    for k in range(start, max_end+1):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=k,
                             random_state=42, passes=6, workers=workers)
        cm = CoherenceModel(model=model, texts=texts_tokenized, dictionary=dictionary, coherence='c_v')
        score = cm.get_coherence()
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    return best_model, dictionary, corpus, best_k

def run_subtopic_modeling(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if 'topic_id' not in df.columns:
        # allow both 'topic_id' and 'topic' names from earlier scripts
        if 'topic' in df.columns:
            df['topic_id'] = df['topic']
        else:
            raise KeyError("Input CSV must contain 'topic_id' column (run topic_modeling.py first).")

    # comment text col (from preprocess)
    if 'comment_text_lem' not in df.columns:
        # try fallback columns
        if 'comment_body' in df.columns:
            df['comment_text_lem'] = df['comment_body'].fillna("").astype(str)
        else:
            raise KeyError("Input CSV must contain 'comment_text_lem' or 'comment_body' column.")

    df['comment_text_lem'] = df['comment_text_lem'].fillna("").astype(str)

    # prepare output column
    df['comment_subtopic'] = -1

    main_topics = sorted([t for t in df['topic_id'].dropna().unique() if t != -1])
    print(f"Found {len(main_topics)} main topics: {main_topics}")

    for mt in main_topics:
        print(f"\n===== Main Topic {mt} =====")

        subset_idx = df[df['topic_id'] == mt].index
        subset = df.loc[subset_idx].copy()

        # keep only non-empty comments
        subset = subset[subset['comment_text_lem'].str.strip() != ""]
        if len(subset) < MIN_DOCS_PER_TOPIC:
            print(f"  ⚠️ Skipping topic {mt}: only {len(subset)} comments (< {MIN_DOCS_PER_TOPIC})")
            continue

        # clean + tokenize comments
        cleaned = [clean_text(t) for t in subset['comment_text_lem'].tolist()]
        tokenized = tokenize_docs(cleaned)
        # filter out empties
        tokenized_filtered = [t for t in tokenized if len(t) > 0]
        if len(tokenized_filtered) < MIN_DOCS_PER_TOPIC:
            print(f"  ⚠️ Skipping topic {mt}: not enough tokenized comments after filtering")
            continue

        tokenized_bi = make_bigrams_tokenized(tokenized_filtered, min_count=3, threshold=8)

        # find optimal subtopic count
        lda_model, dictionary, corpus, best_k = find_best_k_for_texts(tokenized_bi, start=SUB_MIN_K, end=SUB_MAX_K, workers=WORKERS)

        if lda_model is None or best_k is None or best_k == 0:
            print(f"  ⚠️ Could not fit LDA for topic {mt}")
            continue

        print(f"  ✅ main_topic={mt} best_k={best_k}")

        # get dominant subtopic per comment (use model topic ids)
        subtopic_ids = []
        for doc_bow in corpus:
            doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
            if not doc_topics:
                subtopic_ids.append(-1)
            else:
                subtopic_ids.append(int(max(doc_topics, key=lambda x: x[1])[0]))

        # write subtopic ids back to the correct rows in df
        # Note: tokenized_filtered corresponds to subset rows after dropping empty tokens.
        # We must map them back carefully. We'll rebuild mapping using index positions.
        filtered_indices = [i for i, t in enumerate(tokenized) if len(t) > 0]
        # tokenized_filtered[i] corresponds to subset.iloc[filtered_indices[i]]
        for local_pos, sid in enumerate(subtopic_ids):
            row_pos = subset.index[filtered_indices[local_pos]]
            df.at[row_pos, 'comment_subtopic'] = sid

        # print top words for subtopics for quick inspection
        for tid in range(best_k):
            words = lda_model.show_topic(tid, topn=8)
            words_str = ", ".join([w for w, _ in words])
            print(f"    Subtopic {tid}: {words_str}")

    # save final dataframe
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved subtopic-annotated data to {output_csv}")
    return df

if __name__ == "__main__":
    run_subtopic_modeling()
