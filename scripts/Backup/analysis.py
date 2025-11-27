# scripts/analysis.py

import pandas as pd


topic_labels = {
    1: "Cycle & Hormonal Adjustments",
    2: "Doctor Consultations & HRT",
    3: "Life Reflections during Menopause",
    4: "Sleep & Hot Flash Issues",
    5: "Anxiety & Pain Experiences",
    6: "Medical Research & Risk Awareness",
    7: "Intimacy & Relationship Concernss",
}


df = pd.read_csv("data/processed/menopause_with_sentiment.csv")

df["topic_label"] = df["topic_id"].map(topic_labels)

df.to_csv("data/processed/menopause_final_labeled.csv", index=False)
print("✅ Saved final labeled dataset.")

summary = (
    df.groupby("topic_label")["sentiment_label"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    * 100
)
print(summary.round(1))

print("\nAverage Sentiment (Compound) by Topic:")
sent_means = df.groupby("topic_label")["sentiment_compound"].mean().sort_values(ascending=False)
print(sent_means.round(3))


import matplotlib.pyplot as plt

summary.plot(kind="bar", stacked=True, figsize=(9,5), colormap="coolwarm")
plt.title("Sentiment Distribution Across Topics")
plt.xlabel("Topic")
plt.ylabel("Percentage of Posts")
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


sent_means.plot(kind="bar", color="teal", figsize=(9,4))
plt.title("Average Sentiment (Compound) by Topic")
plt.ylabel("Mean Sentiment (−1 to +1)")
plt.axhline(0, color="gray", linestyle="--")
plt.tight_layout()
plt.show()



from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define custom sizes (in pixels) for each topic
topic_sizes = {
    "Cycle & Hormonal Adjustments": (450, 500),
    "Doctor Consultations & HRT": (1200, 300),
    "Life Reflections during Menopause": (700, 400),
    "Sleep & Hot Flash Issues": (700, 400),
    "Anxiety & Pain Experiences": (700, 400),
    "Medical Research & Risk Awareness": (700, 400),
    "Intimacy & Relationship Concernss": (700, 400)
}

for topic in df["topic_label"].unique():
    text = " ".join(df.loc[df["topic_label"] == topic, "text_lem"])
    w, h = topic_sizes.get(topic, (800, 400))  # default fallback size

    wc = WordCloud(
        width=w,
        height=h,
        background_color=None,
        mode="RGBA",
        colormap="PuRd",   # soft pink tone
        max_words=100
    ).generate(text)

    plt.figure(figsize=(w/100, h/100))  # convert px → inches (100 DPI)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"outputs/figures/wordcloud - {topic.replace('&','and').replace(' ','_').replace('/', "")}.png", 
                dpi=300, transparent=True)
    plt.close()
