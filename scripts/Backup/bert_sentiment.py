"""
BERT (HuggingFace) sentiment analysis script for Reddit data.
- Reads a processed CSV with a text column (text_raw or text_lem)
- Predicts sentiment using a pretrained transformer
- Saves outputs (bert_score, bert_label) to a CSV
- Optionally compares with VADER (requires sentiment_label column already present)
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Transformers imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# For evaluation
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

# ---------- Config ----------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  # good general social-media model
# alternatives:
# MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
# ----------------------------

def load_model(model_name=MODEL_NAME, device=-1):
    """
    Load tokenizer and model, return a pipeline.
    device: -1 for CPU, 0..N for GPU index
    """
    print("Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # create pipeline
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    return nlp

def map_label_cardiff(result):
    """
    cardiffnlp returns labels like 'LABEL_0','LABEL_1','LABEL_2' corresponding to negative/neutral/positive.
    We map to ['negative','neutral','positive'] by probability order used for this model.
    """
    lab = result['label']
    # For this model mapping is: LABEL_0 => negative, LABEL_1 => neutral, LABEL_2 => positive
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return mapping.get(lab, lab)

def predict_batch(predictor, texts, batch_size=32, max_length=256):
    """
    Run predictions in batches. Truncates texts to max_length tokens via tokenizer behavior in pipeline.
    Returns list of dicts: [{'label':..., 'score':...}, ...]
    """
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
        batch = texts[i:i+batch_size]
        # pipeline handles truncation; we just call it
        preds = predictor(batch)
        # predictor returns a list of dicts
        results.extend(preds)
    return results

def majority_map_3_to_3(label):
    # no-op placeholder if using model with 3 classes
    return label

def run(args):
    # load csv
    df = pd.read_csv(args.input_csv)
    text_col = args.text_col
    if text_col not in df.columns:
        raise ValueError(f"Text column {text_col} not found in {args.input_csv}. Available columns: {list(df.columns)}")

    device = 0 if (torch.cuda.is_available() and not args.force_cpu) else -1
    if device == 0:
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")

    predictor = load_model(MODEL_NAME, device=device)

    texts = df[text_col].fillna("").astype(str).tolist()
    preds = predict_batch(predictor, texts, batch_size=args.batch_size)

    # normalize outputs to label + score
    labels = []
    scores = []
    for p in preds:
        # model-specific mapping
        if MODEL_NAME.startswith("cardiffnlp"):
            lbl = map_label_cardiff(p)
            sc = float(p["score"])
        else:
            # For other models, the pipeline returns 'label' as human-readable, e.g. "POSITIVE" or "NEGATIVE"
            lbl = p["label"].lower()
            sc = float(p["score"])
        labels.append(lbl)
        scores.append(sc)

    df["bert_label"] = labels
    df["bert_score"] = scores

    # save outputs
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved BERT predictions to:", args.out_csv)

    # If user supplied VADER column (sentiment_label), do a quick comparison
    if "sentiment_label" in df.columns:
        y_vader = df["sentiment_label"].astype(str).tolist()
        y_bert = df["bert_label"].astype(str).tolist()
        # align label names if needed (bert: negative/neutral/positive)
        # compute stats
        acc = np.mean([1 if a==b else 0 for a,b in zip(y_vader, y_bert)])
        print(f"\nAgreement (exact match) between VADER and BERT: {acc*100:.2f}%")
        print("\nClassification report (BERT vs VADER as reference):")
        print(classification_report(y_vader, y_bert, digits=3))
        print("\nCohen's kappa:", cohen_kappa_score(y_vader, y_bert))
        print("\nConfusion matrix (rows=vader, cols=bert):")
        labels_order = ["negative","neutral","positive"]
        cm = confusion_matrix(y_vader, y_bert, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        print(cm_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT sentiment on CSV")
    parser.add_argument("--input_csv", default="data/processed/menopause_full_combined.csv",
                        help="Input CSV (must contain text column)")
    parser.add_argument("--text_col", default="text_lem", help="Text column to use (text_lem or text_clean or text_raw)")
    parser.add_argument("--out_csv", default="data/processed/menopause_with_bert.csv", help="Output CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()
    run(args)
# If new, add it to our set
        processed_post_ids.add(post.id)
        posts_saved_in_batch += 1

        print(f"\nProcessing Post (ID: {post.id}, r/{post.subreddit.display_name})")
        print(f"Title: {post.title[:70]}...")

        # This is a crucial step!
        # It tells PRAW to go and fetch *all* comments.
        try:
            post.comments.replace_more(limit=None)
        except Exception as e:
            print(f"Could not get all comments for post {post.id}: {e}")
            continue # Skip this post if comments fail

        comment_count_for_this_post = 0

        # post.comments.list() gives us a flat list of all comments
        for comment in post.comments.list():
            # We only want actual comments, which have a 'body' attribute.
            if not hasattr(comment, 'body'):
                continue

            # Write all the data to our CSV file
            writer.writerow([
                post.subreddit.display_name,
                post.id,
                post.created_utc,
                post.title,
                post.selftext,
                post.score,
                post.permalink,
                comment.id,
                comment.body,
                comment.score,
                comment.created_utc
            ])
            comment_count_for_this_post += 1

        print(f"Saved {comment_count_for_this_post} comments from this post.")
        comments_saved_in_batch += comment_count_for_this_post