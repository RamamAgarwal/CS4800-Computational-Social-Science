#!/usr/bin/env python3
"""
Exploratory Data Analysis helper for menopause Reddit data.

Builds on the existing pipeline outputs:
  - preprocessing.py -> topic_modelling.py -> sub_topic_modelling.py -> sentiment_analysis.py
  - expected input: data/processed/processed_sentiment_analysis.csv

It generates plots to answer recurrent research questions observed in prior
runs (see outputs/figures/eda_summary.txt):
1) How does sentiment shift over time (BERT vs VADER)?
2) How does topic prevalence change over time?
3) Which topics/subtopics carry disproportionate negativity?
4) Where do BERT and VADER disagree most?
5) Do activity spikes coincide with sentiment swings?

Usage:
  python scripts/eda_questions.py --input data/processed/processed_sentiment_analysis.csv
Outputs will be written under outputs/eda/.
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ----------------------------
# Small helpers
# ----------------------------
def safe_get_col(df: pd.DataFrame, candidates) -> Optional[str]:
    """Return the first matching column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_datetime(col_series: pd.Series) -> pd.Series:
    """Coerce a series to datetime, handling unix seconds if needed."""
    ser = col_series.dropna()
    if ser.empty:
        return pd.Series([], dtype="datetime64[ns]")
    sample = ser.iloc[0]
    try:
        if isinstance(sample, (int, float, np.integer, np.floating)):
            return pd.to_datetime(col_series, unit="s", errors="coerce")
        return pd.to_datetime(col_series, errors="coerce")
    except Exception:
        return pd.to_datetime(col_series, errors="coerce")


def save_plot(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------
# EDA plotters
# ----------------------------
def sentiment_over_time(df: pd.DataFrame, outdir: str) -> Optional[pd.DataFrame]:
    """Line plots of sentiment proportions over time (monthly)."""
    time_col = safe_get_col(
        df,
        [
            "comment_created_utc",
            "post_created_utc",
            "created_utc",
            "created",
            "timestamp",
        ],
    )
    if time_col is None:
        print("No timestamp column found; skipping sentiment_over_time.")
        return None

    ts = ensure_datetime(df[time_col])
    df = df.assign(_time=ts)
    df = df[~df["_time"].isna()].copy()
    if df.empty:
        print("No valid timestamps after parsing; skipping sentiment_over_time.")
        return None

    df["_month"] = df["_time"].dt.to_period("M").dt.to_timestamp()

    results = {}
    for model_col, model_name in [
        (safe_get_col(df, ["bert_label", "hf_label"]), "BERT"),
        (safe_get_col(df, ["vader_label"]), "VADER"),
    ]:
        if model_col is None:
            print(f"No {model_name} labels; skipping.")
            continue
        grp = df.groupby(["_month", model_col]).size().unstack(fill_value=0)
        grp_pct = grp.divide(grp.sum(axis=1), axis=0).reset_index()
        results[model_name] = grp_pct

        fig, ax = plt.subplots(figsize=(9, 4))
        for lab in ["positive", "neutral", "negative"]:
            if lab in grp_pct.columns:
                ax.plot(grp_pct["_month"], grp_pct[lab], marker="o", label=lab)
        ax.set_title(f"Sentiment over time ({model_name})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Proportion")
        ax.legend()
        save_plot(fig, os.path.join(outdir, f"sentiment_over_time_{model_name.lower()}.png"))

    return results.get("BERT") or results.get("VADER")


def topic_prevalence_over_time(
    df: pd.DataFrame, topic_col: Optional[str], outdir: str
) -> Optional[pd.DataFrame]:
    """Stacked area of topic prevalence over time."""
    if topic_col is None:
        print("No topic column found; skipping topic prevalence.")
        return None

    time_col = safe_get_col(df, ["comment_created_utc", "post_created_utc", "created_utc", "created"])
    if time_col is None:
        print("No timestamp column found; skipping topic prevalence over time.")
        return None

    ts = ensure_datetime(df[time_col])
    df = df.assign(_time=ts)
    df = df[~df["_time"].isna()].copy()
    if df.empty:
        print("No valid timestamps after parsing; skipping topic prevalence.")
        return None

    df["_month"] = df["_time"].dt.to_period("M").dt.to_timestamp()
    grp = df.groupby(["_month", topic_col]).size().unstack(fill_value=0)
    grp_pct = grp.divide(grp.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    grp_pct.plot.area(ax=ax, cmap="tab20")
    ax.set_title("Topic prevalence over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Proportion of posts/comments")
    save_plot(fig, os.path.join(outdir, "topic_prevalence_over_time.png"))
    return grp_pct


def topic_sentiment_heatmap(
    df: pd.DataFrame,
    topic_col: Optional[str],
    subtopic_col: Optional[str],
    outdir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Heatmaps for sentiment distribution per topic and subtopic."""
    os.makedirs(outdir, exist_ok=True)
    vader_col = safe_get_col(df, ["vader_label"])
    bert_col = safe_get_col(df, ["bert_label", "hf_label"])
    model_cols = [(vader_col, "VADER"), (bert_col, "BERT")]

    def _plot_heatmap(pivot: pd.DataFrame, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title(title)
        save_plot(fig, os.path.join(outdir, fname))

    topic_df = None
    subtopic_df = None

    for col, name in model_cols:
        if col is None:
            continue
        if topic_col:
            agg = df.groupby([topic_col, col]).size().unstack(fill_value=0)
            agg = agg.divide(agg.sum(axis=1), axis=0)
            topic_df = agg
            _plot_heatmap(
                agg,
                f"{name} sentiment by topic",
                f"topic_sentiment_heatmap_{name.lower()}.png",
            )
        if subtopic_col:
            agg = df.groupby([subtopic_col, col]).size().unstack(fill_value=0)
            agg = agg.divide(agg.sum(axis=1), axis=0)
            subtopic_df = agg
            _plot_heatmap(
                agg,
                f"{name} sentiment by subtopic",
                f"subtopic_sentiment_heatmap_{name.lower()}.png",
            )

    return topic_df, subtopic_df


def model_disagreement(df: pd.DataFrame, outdir: str) -> Optional[pd.DataFrame]:
    """Confusion matrix of BERT vs VADER labels."""
    bert_col = safe_get_col(df, ["bert_label", "hf_label"])
    vader_col = safe_get_col(df, ["vader_label"])
    if bert_col is None or vader_col is None:
        print("Missing bert/vader labels; skipping disagreement matrix.")
        return None

    cm = pd.crosstab(df[bert_col], df[vader_col], normalize="all")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples", ax=ax)
    ax.set_title("BERT vs VADER disagreement (normalized)")
    ax.set_xlabel("VADER")
    ax.set_ylabel("BERT")
    save_plot(fig, os.path.join(outdir, "bert_vs_vader_confusion.png"))
    return cm


def volume_vs_sentiment(df: pd.DataFrame, outdir: str) -> Optional[pd.DataFrame]:
    """Scatter/reg plot: monthly volume vs negative sentiment share (BERT)."""
    time_col = safe_get_col(df, ["comment_created_utc", "post_created_utc", "created_utc", "created"])
    bert_col = safe_get_col(df, ["bert_label", "hf_label"])
    if time_col is None or bert_col is None:
        print("Need timestamp and bert_label for volume_vs_sentiment; skipping.")
        return None

    ts = ensure_datetime(df[time_col])
    df = df.assign(_time=ts)
    df = df[~df["_time"].isna()].copy()
    if df.empty:
        print("No valid timestamps after parsing; skipping volume_vs_sentiment.")
        return None

    df["_month"] = df["_time"].dt.to_period("M").dt.to_timestamp()
    agg = df.groupby("_month").agg(
        volume=("bert_label", "count"),
        neg_share=("bert_label", lambda s: (s == "negative").mean()),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=agg, x="volume", y="neg_share", ax=ax)
    ax.set_title("Does higher activity relate to negativity? (BERT)")
    ax.set_xlabel("Posts/comments per month")
    ax.set_ylabel("Negative share")
    save_plot(fig, os.path.join(outdir, "volume_vs_negative_share.png"))
    return agg


# ----------------------------
# Main
# ----------------------------
def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    outdir = os.path.join("outputs", "eda")
    os.makedirs(outdir, exist_ok=True)

    topic_col = safe_get_col(df, ["topic_id", "main_topic", "topic"])
    subtopic_col = safe_get_col(df, ["comment_subtopic", "subtopic"])

    print("1) Sentiment over time")
    sentiment_over_time(df, outdir)

    print("2) Topic prevalence over time")
    topic_prevalence_over_time(df, topic_col, outdir)

    print("3) Topic/Subtopic sentiment heatmaps")
    topic_sentiment_heatmap(df, topic_col, subtopic_col, outdir)

    print("4) BERT vs VADER disagreement")
    model_disagreement(df, outdir)

    print("5) Activity vs negativity")
    volume_vs_sentiment(df, outdir)

    print(f"\nDone. Plots saved to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "../"
        default="data/processed/processed_sentiment_analysis.csv",
        help="CSV with sentiment + topic columns (output of sentiment_analysis.py)",
    )
    args = parser.parse_args()
    main(args)

