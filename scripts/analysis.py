#!/usr/bin/env python3
"""
Analysis script (robust plotting)

Produces:
  - topic-level sentiment CSV + stacked-bar PNGs (VADER & BERT)
  - subtopic-level sentiment CSV + per-topic stacked-bar PNGs
  - sentiment shift over years CSV + line PNGs
  - posting activity histogram PNG + top users CSV + top-users bar PNG

Usage:
  python scripts/analysis.py --input path/to/menopause_with_sentiment.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
# Use a non-interactive backend that's safe on servers/windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

# ----------------------------
# Helpers
# ----------------------------
def safe_get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_datetime(col_series):
    ser = col_series.dropna()
    if ser.empty:
        return pd.Series([], dtype="datetime64[ns]")
    sample = ser.iloc[0]
    try:
        if isinstance(sample, (int, float, np.integer, np.floating)):
            return pd.to_datetime(col_series, unit='s', errors='coerce')
        return pd.to_datetime(col_series, errors='coerce')
    except Exception:
        return pd.to_datetime(col_series, errors='coerce')

def top_n_words(texts, n=15, stopwords=None):
    stopwords = stopwords or set()
    cnt = Counter()
    for t in texts:
        if not isinstance(t, str): 
            continue
        words = re.findall(r"[A-Za-z']+", t.lower())
        for w in words:
            if len(w) < 3 or w in stopwords:
                continue
            cnt[w] += 1
    return cnt.most_common(n)

def save_plot(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ----------------------------
# Plot helpers
# ----------------------------
def stacked_bar_by_topic(df_agg, model_prefix, outpath):
    """df_agg must contain columns: topic, <prefix>_pos_pct, <prefix>_neu_pct, <prefix>_neg_pct"""
    topics = df_agg["topic"].astype(str).tolist()
    pos = df_agg[f"{model_prefix}_pos_pct"].fillna(0).values
    neu = df_agg[f"{model_prefix}_neu_pct"].fillna(0).values
    neg = df_agg[f"{model_prefix}_neg_pct"].fillna(0).values
    x = np.arange(len(topics))
    fig, ax = plt.subplots(figsize=(max(6, len(topics)*0.6), 4))
    ax.bar(x, pos, label='positive')
    ax.bar(x, neu, bottom=pos, label='neutral')
    ax.bar(x, neg, bottom=pos+neu, label='negative')
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.set_ylabel("Proportion")
    ax.set_title(f"Topic-wise sentiment distribution ({model_prefix})")
    ax.legend()
    save_plot(fig, outpath)
    print(f"Saved {outpath}")

def stacked_bar_subtopics_per_topic(subtopic_df, model_prefix, outdir):
    """Create one stacked bar per main topic showing subtopic sentiment distribution"""
    os.makedirs(outdir, exist_ok=True)
    if subtopic_df is None or subtopic_df.empty:
        print("No subtopic_df available for plotting.")
        return
    for topic_id, grp in subtopic_df.groupby("topic"):
        grp_sorted = grp.sort_values("subtopic")
        subtopic_labels = grp_sorted["subtopic"].astype(str).tolist()
        pos = grp_sorted[f"{model_prefix}_pos_pct"].fillna(0).values
        neu = grp_sorted[f"{model_prefix}_neu_pct"].fillna(0).values
        neg = grp_sorted[f"{model_prefix}_neg_pct"].fillna(0).values
        x = np.arange(len(subtopic_labels))
        fig, ax = plt.subplots(figsize=(max(6, len(subtopic_labels)*0.6), 4))
        ax.bar(x, pos, label='positive')
        ax.bar(x, neu, bottom=pos, label='neutral')
        ax.bar(x, neg, bottom=pos+neu, label='negative')
        ax.set_xticks(x)
        ax.set_xticklabels(subtopic_labels, rotation=45, ha='right')
        ax.set_ylabel("Proportion")
        ax.set_title(f"Topic {topic_id} — Subtopic sentiment ({model_prefix})")
        ax.legend()
        outpath = os.path.join(outdir, f"topic_{topic_id}_subtopic_sentiment_{model_prefix.lower()}.png")
        save_plot(fig, outpath)
        print(f"Saved {outpath}")

# ----------------------------
# Main analysis functions
# ----------------------------
def topic_sentiment_distribution(df, topic_col, subtopic_col, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vader_col = safe_get_col(df, ["vader_label", "vader_sentiment", "vader_polarity"])
    bert_col = safe_get_col(df, ["bert_label", "hf_label", "bert_sentiment"])

    if vader_col is None and bert_col is None:
        raise KeyError("No sentiment label column found (vader_label or bert_label).")

    # topic-level aggregation
    topic_vals = df[topic_col].fillna("None") if topic_col in df.columns else pd.Series(["None"]*len(df))
    g = df.copy()
    g['__topic'] = topic_vals
    agg_rows = []
    for topic, grp in g.groupby("__topic"):
        row = {"topic": topic, "n_rows": len(grp)}
        if vader_col:
            vc = grp[vader_col].value_counts(normalize=True)
            row.update({
                "vader_pos_pct": float(vc.get("positive", 0.0)),
                "vader_neu_pct": float(vc.get("neutral", 0.0)),
                "vader_neg_pct": float(vc.get("negative", 0.0))
            })
        if bert_col:
            bc = grp[bert_col].value_counts(normalize=True)
            row.update({
                "bert_pos_pct": float(bc.get("positive", 0.0)),
                "bert_neu_pct": float(bc.get("neutral", 0.0)),
                "bert_neg_pct": float(bc.get("negative", 0.0))
            })
        agg_rows.append(row)
    topic_df = pd.DataFrame(agg_rows).sort_values("topic")
    topic_df.to_csv(os.path.join(out_dir, "topic_level_sentiment_distribution.csv"), index=False)
    print(f"Saved topic-level sentiment CSV to {os.path.join(out_dir, 'topic_level_sentiment_distribution.csv')}")

    # plot VADER/BERT stacked bars if available
    if vader_col:
        stacked_bar_by_topic(topic_df, "vader", os.path.join(out_dir, "topic_sentiment_vader.png"))
    if bert_col:
        stacked_bar_by_topic(topic_df, "bert", os.path.join(out_dir, "topic_sentiment_bert.png"))

    # Subtopic-level aggregation
    subtopic_df = None
    if subtopic_col in df.columns:
        agg_rows = []
        gb = df.groupby([topic_col, subtopic_col]) if topic_col in df.columns else df.groupby([subtopic_col])
        for keys, grp in gb:
            if isinstance(keys, tuple):
                t, s = keys
            else:
                t, s = (None, keys)
            row = {"topic": t, "subtopic": s, "n_rows": len(grp)}
            if vader_col:
                vc = grp[vader_col].value_counts(normalize=True)
                row.update({
                    "vader_pos_pct": float(vc.get("positive", 0.0)),
                    "vader_neu_pct": float(vc.get("neutral", 0.0)),
                    "vader_neg_pct": float(vc.get("negative", 0.0))
                })
            if bert_col:
                bc = grp[bert_col].value_counts(normalize=True)
                row.update({
                    "bert_pos_pct": float(bc.get("positive", 0.0)),
                    "bert_neu_pct": float(bc.get("neutral", 0.0)),
                    "bert_neg_pct": float(bc.get("negative", 0.0))
                })
            agg_rows.append(row)
        subtopic_df = pd.DataFrame(agg_rows).sort_values(["topic","subtopic"])
        subtopic_df.to_csv(os.path.join(out_dir, "subtopic_level_sentiment_distribution.csv"), index=False)
        print(f"Saved subtopic-level sentiment CSV to {os.path.join(out_dir, 'subtopic_level_sentiment_distribution.csv')}")

        # Plot per-topic subtopic sentiment
        if vader_col:
            stacked_bar_subtopics_per_topic(subtopic_df, "vader", out_dir)
        if bert_col:
            stacked_bar_subtopics_per_topic(subtopic_df, "bert", out_dir)

    return topic_df, subtopic_df

def sentiment_shift_over_years(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    time_col = safe_get_col(df, ["comment_created_utc", "post_created_utc", "created_utc", "created"])
    if time_col is None:
        print("No timestamp column found; skipping sentiment shift over years.")
        return None

    times = ensure_datetime(df[time_col])
    df['_time'] = times
    df = df[~df['_time'].isna()].copy()
    if df.empty:
        print("No valid timestamps after parsing; skipping sentiment shift.")
        return None
    df['year'] = df['_time'].dt.year

    vader_col = safe_get_col(df, ["vader_label"])
    bert_col = safe_get_col(df, ["bert_label"])

    out = {}
    for model_col, model_name in [(vader_col, "VADER"), (bert_col, "BERT")]:
        if model_col is None:
            print(f"No {model_name} column found; skipping.")
            continue
        grp = df.groupby(['year', model_col]).size().unstack(fill_value=0)
        grp_pct = grp.divide(grp.sum(axis=1), axis=0).reset_index()
        fname = os.path.join(out_dir, f"sentiment_over_years_{model_name.lower()}.csv")
        grp_pct.to_csv(fname, index=False)
        print(f"Saved {fname}")

        # plot
        fig, ax = plt.subplots(figsize=(8,4))
        for label in ['positive','neutral','negative']:
            if label in grp_pct.columns:
                ax.plot(grp_pct['year'], grp_pct[label], marker='o', label=label)
        ax.set_title(f"Sentiment shift over years ({model_name})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion")
        ax.legend()
        outpath = os.path.join(out_dir, f"sentiment_over_years_{model_name.lower()}.png")
        save_plot(fig, outpath)
        print(f"Saved {outpath}")
        out[model_name] = grp_pct
    return out

def posting_activity_and_top_users(df, text_col, user_col, topic_col, subtopic_col, out_dir, top_n=20):
    os.makedirs(out_dir, exist_ok=True)
    empty_user_counts = pd.DataFrame(columns=["user", "n_posts"])
    empty_top_users = pd.DataFrame(columns=["user", "n_posts", "topic_dist", "subtopic_dist", "top_words"])
    if user_col is None:
        print("No user column found; skipping posting activity/top users analysis.")
        # save empty csvs for consistency
        empty_user_counts.to_csv(os.path.join(out_dir, "user_post_counts.csv"), index=False)
        empty_top_users.to_csv(os.path.join(out_dir, "top_users_topic_subtopic_words.csv"), index=False)
        return empty_user_counts, empty_top_users

    user_counts = df[user_col].fillna("UNKNOWN").value_counts().rename_axis("user").reset_index(name="n_posts")
    user_counts.to_csv(os.path.join(out_dir, "user_post_counts.csv"), index=False)
    print(f"Saved {os.path.join(out_dir, 'user_post_counts.csv')}")

    # histogram with log y if distribution heavy-tailed
    try:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(user_counts['n_posts'], bins=50, log_scale=(False, True), ax=ax)
        ax.set_xlabel("Posts/comments by user")
        ax.set_ylabel("Number of users (log scale)")
        ax.set_title("Histogram: posting activity (#posts per user)")
        outpath = os.path.join(out_dir, "posting_activity_histogram.png")
        save_plot(fig, outpath)
        print(f"Saved {outpath}")
    except Exception as e:
        print("Warning: failed to generate posting activity histogram:", e)
        # fallback simpler histogram
        try:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(user_counts['n_posts'], bins=50)
            ax.set_xlabel("Posts/comments by user")
            ax.set_ylabel("Number of users")
            ax.set_title("Histogram: posting activity (#posts per user)")
            outpath = os.path.join(out_dir, "posting_activity_histogram_fallback.png")
            save_plot(fig, outpath)
            print(f"Saved fallback {outpath}")
        except Exception as e2:
            print("Failed fallback histogram as well:", e2)

    # Top users analysis
    top_users = user_counts.head(top_n)['user'].tolist()
    rows = []
    stopwords = set()
    text_source = text_col if (text_col in df.columns) else None

    for u in tqdm(top_users, desc="Top users analysis"):
        user_df = df[df[user_col] == u]
        n = len(user_df)
        topic_dist = user_df[topic_col].value_counts(normalize=True).to_dict() if (topic_col is not None and topic_col in df.columns) else {}
        subtopic_dist = user_df[subtopic_col].value_counts(normalize=True).to_dict() if (subtopic_col is not None and subtopic_col in df.columns) else {}
        texts = user_df[text_source].fillna("").astype(str).tolist() if text_source else []
        top_words = top_n_words(texts, n=20, stopwords=stopwords)
        rows.append({
            "user": u,
            "n_posts": n,
            "topic_dist": topic_dist,
            "subtopic_dist": subtopic_dist,
            "top_words": ";".join([f"{w}:{c}" for w,c in top_words])
        })

    top_users_df = pd.DataFrame(rows)
    top_users_df.to_csv(os.path.join(out_dir, "top_users_topic_subtopic_words.csv"), index=False)
    print(f"Saved {os.path.join(out_dir, 'top_users_topic_subtopic_words.csv')}")

    # bar plot top users by post count
    try:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(data=user_counts.head(20), x='user', y='n_posts', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Top 20 users by post/comment count")
        outpath = os.path.join(out_dir, "top20_users_bar.png")
        save_plot(fig, outpath)
        print(f"Saved {outpath}")
    except Exception as e:
        print("Warning: failed to plot top users bar:", e)

    return user_counts, top_users_df

# ----------------------------
# Orchestrator
# ----------------------------
def main(args):
    input_candidates = [
        "data/processed/processed_sentiment_analysis.csv"
    ]
    input_file = None
    for c in input_candidates:
        if c and os.path.exists(c):
            input_file = c
            break
    if input_file is None:
        raise FileNotFoundError("No input CSV found. Provide --input or ensure processed CSVs exist.")

    print(f"Loading {input_file}")
    df = pd.read_csv(input_file)
    print(f"Rows: {len(df)}")

    topic_col = safe_get_col(df, ["topic_id", "main_topic", "topic"])
    subtopic_col = safe_get_col(df, ["comment_subtopic", "subtopic", "comment_subtopic_id"])
    text_col = safe_get_col(df, ["sentiment_text", "comment_text_clean", "comment_text_lem", "post_text_clean", "post_text_lem", "text_clean", "text_lem"])
    user_col = safe_get_col(df, ["author", "username", "user", "user_name"])

    out_summary_dir = "data/processed/summaries"
    out_plot_dir = "outputs"

    print("1) Topic & Subtopic sentiment distributions")
    try:
        topic_df, subtopic_df = topic_sentiment_distribution(df, topic_col, subtopic_col, out_summary_dir)
    except Exception as e:
        print("Error during topic sentiment distribution:", e)
        topic_df, subtopic_df = None, None

    print("2) Sentiment shift over years")
    try:
        years_out = sentiment_shift_over_years(df, out_plot_dir)
    except Exception as e:
        print("Error during sentiment shift over years:", e)
        years_out = None

    print("3) Posting activity histogram & top users")
    try:
        user_counts, top_users_df = posting_activity_and_top_users(df, text_col, user_col, topic_col, subtopic_col, out_summary_dir, top_n=30)
    except Exception as e:
        print("Error during posting activity analysis:", e)
        user_counts, top_users_df = pd.DataFrame(), pd.DataFrame()

    print("\nAll outputs saved to:")
    print(" - summary CSVs:", out_summary_dir)
    print(" - plots:", out_plot_dir)
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to processed CSV with sentiment & topic columns.")
    args = parser.parse_args()
    main(args)
