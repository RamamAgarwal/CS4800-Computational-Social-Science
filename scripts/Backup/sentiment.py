# scripts/sentiment.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def compute_sentiment(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    df['sentiment_compound'] = df['text_raw'].fillna("").apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_label'] = df['sentiment_compound'].apply(lambda s: "positive" if s>=0.05 else ("negative" if s<=-0.05 else "neutral"))
    df.to_csv(out_csv, index=False)
    print("Saved with sentiment:", out_csv)
    # quick plot
    sns.countplot(data=df, x='sentiment_label')
    plt.title('Sentiment distribution')
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inpath", default="data/processed/menopause_with_topics.csv")
    parser.add_argument("--out", dest="outpath", default="data/processed/menopause_with_sentiment.csv")
    args = parser.parse_args()
    compute_sentiment(args.inpath, args.outpath)
