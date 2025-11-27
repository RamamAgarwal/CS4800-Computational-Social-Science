# scripts/preprocess.py
import pandas as pd
import re
import spacy
from tqdm import tqdm
import os, emoji

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# URL_RE = re.compile(r'http\S+|www\.\S+')
# EMOJI_RE = re.compile("["
#     u"\U0001F600-\U0001F64F"  # emoticons
#     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#     u"\U0001F680-\U0001F6FF"  # transport & map symbols
#     u"\U0001F1E0-\U0001F1FF"  # flags
# "]+", flags=re.UNICODE)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove usernames
    text = re.sub(r"u\/[A-Za-z0-9_]+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # Remove phone numbers
    text = re.sub(r"\b\d{10}\b", "", text)
    # Remove mentions like Dr. or Mr./Ms. followed by a name
    text = re.sub(r"(dr|mr|ms|mrs)\.?\s+[A-Z][a-z]+", "", text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    # Remove non-standard characters
    text = text.replace("\n", " ").replace("\r", " ")

    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)  # keep only normal characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def lemmatize_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def preprocess_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    print(f"🔹 Loaded {len(df)} posts from {input_csv}")
    tqdm.pandas()

    # Combine title + body if both exist
    if "title" in df.columns and "body" in df.columns:
        df["text_raw"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    else:
        df["text_raw"] = df["body"].fillna("")

    df["text_clean"] = df["text_raw"].progress_apply(clean_text)
    df["text_lem"] = df["text_clean"].progress_apply(lemmatize_text)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"✅ Saved preprocessed data → {output_csv}")

if __name__ == "__main__":
    input_csv = "data/raw/menopause_reddit_praw.csv"
    output_csv = "data/processed/menopause_processed.csv"
    preprocess_csv(input_csv, output_csv)
