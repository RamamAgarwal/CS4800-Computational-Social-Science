# scripts/reddit_scraper.py
import praw
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

def init_reddit():
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
    return reddit

def fetch_posts(subreddits, query, limit=500):
    reddit = init_reddit()
    data = []

    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        print(f"Fetching posts from r/{sub} for query: '{query}'")
        for post in tqdm(subreddit.search(query, limit=limit, sort="new")):
            data.append({
                "id": post.id,
                "subreddit": sub,
                "title": post.title,
                "body": post.selftext,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "url": post.url
            })
    return pd.DataFrame(data)

if __name__ == "__main__":
    subreddits = ["Menopause", "perimenopause", "earlymenopause", "menopausesupport"]
    queries = ["menopause", "perimenopause", "hot flashes", "hormone therapy"]

    for i in subreddits:
        print(f"r/{i}")

    # all_data = pd.DataFrame()
    # for q in queries:
    #     df = fetch_posts(subreddits, q, limit=10000)
    #     all_data = pd.concat([all_data, df])

    # os.makedirs("data/raw", exist_ok=True)
    # out_path = "data/raw/menopause_reddit_praw.csv"
    # all_data.to_csv(out_path, index=False)
    # print(f"✅ Saved {len(all_data)} posts to {out_path}")
