# subreddits = [
#     "Menopause", "Perimenopause", "MenopauseSupport",
#     "EarlyMenopause", "Hormones", "WomensHealth",
#     "TwoXChromosomes", "AskDocs", "Midlife", "HealthyAging"
# ]

# queries = [
#     "menopause", "perimenopause", "hot flashes", "night sweats",
#     "HRT", "hormone therapy", "estrogen", "progesterone",
#     "mood swings", "anxiety", "depression", "brain fog",
#     "sleep issues", "exercise", "diet", "self-care"
# ]


# for i in subreddits:
#     print(f"r/{i}", end=",  ")

# print("\n")


# for q in queries:
#     print(f"'{q}'", end=",  ")

import pandas as pd

data = pd.read_csv("./data/raw/reddit_products_data.csv")

print(data.head())