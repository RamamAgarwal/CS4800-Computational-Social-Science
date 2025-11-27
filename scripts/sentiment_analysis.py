"""
Sentiment Analysis pipeline (VADER vs BERT)

- Input (expected): data/processed/menopause_with_topics_and_subtopics.csv
  Must contain:
    - post_id
    - comment_id (optional)
    - post_text_clean or post_text_lem or post_title/post_body
    - comment_text_clean or comment_text_lem or comment_body
    - topic_id or main_topic
    - comment_subtopic (optional)
- Output:
    - data/processed/menopause_with_sentiment.csv  (row-level: both VADER & BERT scores+labels)
    - data/processed/summaries/
        - post_level_sentiment_by_topic.csv
        - comment_level_sentiment_by_topic_and_subtopic.csv
    - simple PNG summary plots in outputs/
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment libs
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Transformers (BERT-like)
from transformers import pipeline

# Ensure NLTK data
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = "data/processed/processed_subtopic_modeling.csv"
OUTPUT_CSV = "data/processed/processed_sentiment_analysis.csv"
SUMMARY_DIR = "data/processed/summaries"
OUTPUT_DIR = "outputs"

# thresholds for VADER compound -> label
VADER_POS_THRESH = 0.05
VADER_NEG_THRESH = -0.05

# BERT model choice (returns labels: positive/neutral/negative)
# "cardiffnlp/twitter-roberta-base-sentiment" is a good choice with 3 labels.
BERT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

BATCH_SIZE_BERT = 64  # number of texts per batch for transformer pipeline

# ----------------------------
# Helpers
# ----------------------------
def safe_get_col(df, candidates):
    """Return first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def map_vader_label(compound):
    if compound >= VADER_POS_THRESH:
        return "positive"
    elif compound <= VADER_NEG_THRESH:
        return "negative"
    else:
        return "neutral"

def map_bert_label(bert_label):
    """Normalize label strings from some models e.g. 'LABEL_0' handling is needed for some models.
       For cardiffnlp/twitter-roberta-base-sentiment the labels are 'negative','neutral','positive' or similar."""
    s = str(bert_label).lower()
    if "neg" in s:
        return "negative"
    if "pos" in s:
        return "positive"
    if "neu" in s:
        return "neutral"
    # fallback mapping for LABEL_{n}
    if "label_0" in s:
        return "negative"
    if "label_1" in s:
        return "neutral"
    if "label_2" in s:
        return "positive"
    return s

# ----------------------------
# VADER functions
# ----------------------------
def run_vader_on_series(texts):
    """
    texts : list/Series of strings
    returns: DataFrame with columns: vader_compound, vader_pos, vader_neu, vader_neg, vader_label
    """
    analyzer = SentimentIntensityAnalyzer()
    results = {
        "vader_compound": [],
        "vader_pos": [],
        "vader_neu": [],
        "vader_neg": [],
        "vader_label": []
    }

    for t in tqdm(texts, desc="VADER"):
        if not isinstance(t, str) or t.strip() == "":
            vs = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
        else:
            vs = analyzer.polarity_scores(t)
        results["vader_compound"].append(vs["compound"])
        results["vader_pos"].append(vs["pos"])
        results["vader_neu"].append(vs["neu"])
        results["vader_neg"].append(vs["neg"])
        results["vader_label"].append(map_vader_label(vs["compound"]))

    return pd.DataFrame(results)

# ----------------------------
# BERT functions (transformer pipeline)
# ----------------------------
# Add imports near top if not present
from transformers import pipeline, AutoTokenizer

def _safe_model_max_length(tokenizer, default=512, cap=512):
    # model_max_length may be large or a numpy scalar; ensure safe python int and cap it
    try:
        m = getattr(tokenizer, "model_max_length", default)
        m = int(m)
    except Exception:
        m = default
    # if the tokenizer declares an absurdly large max length (some HF tokenizers do),
    # cap to a sensible value to avoid overflow issues in fast tokenizers.
    if m <= 0 or m > cap:
        m = cap
    return m

def build_bert_pipeline(model_name=BERT_MODEL, device=-1):
    print(f"Loading transformer pipeline: {model_name} (this may download model weights)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, device=device, top_k=1)
    return nlp

def run_bert_on_series(texts, nlp_pipeline, batch_size=BATCH_SIZE_BERT):
    """
    Simple truncation-based inference that protects against tokenizer overflow.
    """
    labels = []
    scores = []
    tokenizer = nlp_pipeline.tokenizer
    max_len = _safe_model_max_length(tokenizer, default=512, cap=512)

    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        batch_in = [t if isinstance(t, str) and t.strip() != "" else "neutral" for t in batch]

        # Pass truncation and safe max_length to pipeline
        out = nlp_pipeline(batch_in, truncation=True, max_length=int(max_len))
        for o in out:
            if isinstance(o, list):
                o = o[0]
            lab = map_bert_label(o.get("label", ""))
            sc = float(o.get("score", 0.0))
            labels.append(lab)
            scores.append(sc)

    return pd.DataFrame({"bert_label": labels, "bert_score": scores})


# ----------------------------
# Aggregation helpers
# ----------------------------
def aggregate_and_save(df, outdir=SUMMARY_DIR):
    os.makedirs(outdir, exist_ok=True)

    # Post-level aggregation (if posts exist)
    post_id_col = safe_get_col(df, ["post_id", "postId", "id"])
    if post_id_col:
        posts = df.drop_duplicates(subset=[post_id_col]).copy()
        # prefer post-level text fields; we assume sentiments computed for rows for posts too
        # compute counts by topic
        topic_col = safe_get_col(df, ["topic_id", "main_topic", "topic"])
        if topic_col:
            g = posts.groupby(topic_col).agg(
                total_posts=(post_id_col, "nunique"),
                vader_pos_pct=("vader_label", lambda s: (s=="positive").mean()),
                vader_neu_pct=("vader_label", lambda s: (s=="neutral").mean()),
                vader_neg_pct=("vader_label", lambda s: (s=="negative").mean()),
                bert_pos_pct=("bert_label", lambda s: (s=="positive").mean()),
                bert_neu_pct=("bert_label", lambda s: (s=="neutral").mean()),
                bert_neg_pct=("bert_label", lambda s: (s=="negative").mean()),
                vader_compound_avg=("vader_compound", "mean"),
                bert_score_avg=("bert_score", "mean")
            ).reset_index()
            g.to_csv(os.path.join(outdir, "post_level_sentiment_by_topic.csv"), index=False)
            print(f"Saved post-level sentiment by topic to {outdir}/post_level_sentiment_by_topic.csv")

    # Comment-level aggregation by (topic, subtopic)
    if "comment_subtopic" in df.columns and ("topic_id" in df.columns or "main_topic" in df.columns):
        topic_col = safe_get_col(df, ["topic_id", "main_topic", "topic"])
        agg = df.groupby([topic_col, "comment_subtopic"]).agg(
            n_comments=("comment_id" if "comment_id" in df.columns else "comment_text_lem", "count"),
            vader_pos_pct=("vader_label", lambda s: (s=="positive").mean()),
            vader_neu_pct=("vader_label", lambda s: (s=="neutral").mean()),
            vader_neg_pct=("vader_label", lambda s: (s=="negative").mean()),
            bert_pos_pct=("bert_label", lambda s: (s=="positive").mean()),
            bert_neu_pct=("bert_label", lambda s: (s=="neutral").mean()),
            bert_neg_pct=("bert_label", lambda s: (s=="negative").mean()),
            vader_compound_avg=("vader_compound", "mean"),
            bert_score_avg=("bert_score", "mean")
        ).reset_index()
        agg.to_csv(os.path.join(outdir, "comment_level_sentiment_by_topic_and_subtopic.csv"), index=False)
        print(f"Saved comment-level sentiment by topic+subtopic to {outdir}/comment_level_sentiment_by_topic_and_subtopic.csv")

    # Overall compare distributions (VADER vs BERT)
    overall = {
        "vader_positive_pct": (df["vader_label"]=="positive").mean(),
        "vader_neutral_pct": (df["vader_label"]=="neutral").mean(),
        "vader_negative_pct": (df["vader_label"]=="negative").mean(),
        "bert_positive_pct": (df["bert_label"]=="positive").mean(),
        "bert_neutral_pct": (df["bert_label"]=="neutral").mean(),
        "bert_negative_pct": (df["bert_label"]=="negative").mean(),
        "n_rows": len(df)
    }
    pd.DataFrame([overall]).to_csv(os.path.join(outdir, "overall_sentiment_comparison.csv"), index=False)
    print(f"Saved overall sentiment comparison to {outdir}/overall_sentiment_comparison.csv")

# ----------------------------
# Simple plotting helpers
# ----------------------------
def plot_overall_distributions(df, outdir=OUTPUT_DIR):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    sns.countplot(x="vader_label", data=df, order=["negative","neutral","positive"], ax=axes[0])
    axes[0].set_title("VADER label counts")

    sns.countplot(x="bert_label", data=df, order=["negative","neutral","positive"], ax=axes[1])
    axes[1].set_title("BERT label counts")

    plt.tight_layout()
    out_path = os.path.join(outdir, "sentiment_label_counts.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {out_path}")

# ----------------------------
# Main pipeline
# ----------------------------
def main(args):
    input_csv = args.input
    output_csv = args.output

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"🔹 Loaded {len(df)} rows from {input_csv}")

    # choose text columns (post and comment)
    post_text_col = safe_get_col(df, ["post_text_clean", "post_text_lem", "post_title", "post_body"])
    comment_text_col = safe_get_col(df, ["comment_text_clean", "comment_text_lem", "comment_body", "comment"])

    if post_text_col is None and comment_text_col is None:
        raise KeyError("Couldn't find suitable post or comment text columns. Expected e.g. 'post_text_lem' or 'comment_text_lem'.")

    # We'll compute sentiment rows for both posts and comments per row. If a row is a comment row, comment text will be present.
    # Build a 'sentiment_source_text' column that'll be used for comment-level analysis and 'post_sentiment_text' for post-level.
    # If your CSV is post+comment per row (comments repeated with post info), the row-level text we'll analyze is comment_text if present else post_text.
    df["sentiment_text"] = df[comment_text_col].fillna("").astype(str)
    empty_mask = df["sentiment_text"].str.strip() == ""
    if post_text_col:
        df.loc[empty_mask, "sentiment_text"] = df.loc[empty_mask, post_text_col].fillna("").astype(str)

    # Also compute explicit post-level sentiment on deduplicated posts
    post_id_col = safe_get_col(df, ["post_id", "postId", "id"])

    # --- VADER on all rows ---
    texts_for_vader = df["sentiment_text"].tolist()
    vader_df = run_vader_on_series(texts_for_vader)
    df = pd.concat([df.reset_index(drop=True), vader_df.reset_index(drop=True)], axis=1)

    # --- BERT on all rows ---
    try:
        # Attempt to use GPU if available
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1

    bert_nlp = build_bert_pipeline(model_name=BERT_MODEL, device=device)
    bert_df = run_bert_on_series(df["sentiment_text"].tolist(), bert_nlp, batch_size=BATCH_SIZE_BERT)
    df = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

    # write row-level output
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Row-level sentiment results saved to {output_csv}")

    # Aggregations + summaries
    aggregate_and_save(df, outdir=SUMMARY_DIR)
    plot_overall_distributions(df, outdir=OUTPUT_DIR)
    # print(df)

    print("\nAll done. Check the CSVs and plots in data/processed and outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV (preprocessed + topics + subtopics)")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV to write row-level sentiments")
    args = parser.parse_args()
    main(args)
