# scripts/preprocess.py
import pandas as pd
import re
import spacy
from tqdm import tqdm
import os, emoji

# ⚠️ Make sure you've run:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove Reddit usernames u/username
    text = re.sub(r"u\/[A-Za-z0-9_-]+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove phone numbers (simple 10-digit pattern)
    text = re.sub(r"\b\d{10}\b", "", text)

    # Remove mentions like Dr. or Mr./Ms./Mrs. + Name
    text = re.sub(r"(dr|mr|ms|mrs)\.?\s+[A-Z][a-z]+", "", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Normalize newlines
    text = text.replace("\n", " ").replace("\r", " ")

    # Keep only letters, numbers, basic punctuation
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def lemmatize_series(text_series: pd.Series) -> pd.Series:
    """
    Batched lemmatization for speed.
    - Lowercase
    - Remove stopwords
    - Keep only alphabetic tokens
    """
    text_series = text_series.fillna("").astype(str)
    docs = nlp.pipe(text_series.tolist(), batch_size=64, n_process=2)

    lemmas = []
    for doc in docs:
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        lemmas.append(" ".join(tokens))

    return pd.Series(lemmas, index=text_series.index)


def preprocess_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    print(f"🔹 Loaded {len(df)} rows from {input_csv}")
    tqdm.pandas()

    # --- 1. Build raw text fields ---

    # Post-level text (title + body)
    if {"post_title", "post_body"}.issubset(df.columns):
        df["post_text_raw"] = (
            df["post_title"].fillna("") + " " + df["post_body"].fillna("")
        ).str.strip()
    else:
        # fallback if only body exists
        df["post_text_raw"] = df.get("post_body", "").fillna("")

    # Comment-level text
    if "comment_body" in df.columns:
        df["comment_text_raw"] = df["comment_body"].fillna("")
    else:
        df["comment_text_raw"] = ""

    # Optional: unified text for generic modeling (comment if exists, else post text)
    df["text_raw"] = df["comment_text_raw"]
    empty_mask = df["text_raw"].str.strip() == ""
    df.loc[empty_mask, "text_raw"] = df.loc[empty_mask, "post_text_raw"]

    # --- 2. Clean text ---

    print("Cleaning text ...")
    df["post_text_clean"] = df["post_text_raw"].progress_apply(clean_text)
    df["comment_text_clean"] = df["comment_text_raw"].progress_apply(clean_text)
    df["text_clean"] = df["text_raw"].progress_apply(clean_text)

    # --- 3. Lemmatize text (batched) ---

    print("Lemmatizing post text ...")
    df["post_text_lem"] = lemmatize_series(df["post_text_clean"])

    print("Lemmatizing comment text ...")
    df["comment_text_lem"] = lemmatize_series(df["comment_text_clean"])

    print("Lemmatizing unified text ...")
    df["text_lem"] = lemmatize_series(df["text_clean"])

    # --- 4. Save ---

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved preprocessed data → {output_csv}")


if __name__ == "__main__":
    input_csv = "data/raw/raw_reddit_data.csv"
    output_csv = "data/processed/processed_reddit_data.csv"
    preprocess_csv(input_csv, output_csv)
