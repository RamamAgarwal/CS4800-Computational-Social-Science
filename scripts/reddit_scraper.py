import praw
import csv
import time
import sys
from dotenv import load_dotenv
import os


def init_reddit():
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
    return reddit

# # --- 1. AUTHENTICATION ---
# User's provided credentials - THIS SECTION IS CORRECT
reddit = praw.Reddit(
    client_id="CkqK0rL321nSmK1yVMkobg",
    client_secret="SeEh7FsHpNx4cgXzUrU4lj_G3beNpA",
    username="dunno_who_but",
    password="India@123", # Make sure your password is still here
    user_agent="MenstrualScraper/0.1 by dunno_who_but"
)

# reddit = init_reddit()

# --- 2. CONFIGURATION ---

# New expanded list of subreddits
subreddit_list = ["Menopause", "perimenopause", "earlymenopause", "menopausesupport", "hormones", "HormoneTherapy"]
# Remove duplicates (like 'ftm' and 'FtM') by making all lowercase
subreddit_list = list(set([sub.lower() for sub in subreddit_list]))



queries = [
    "menopause", "perimenopause", "hot flashes", "night sweats",
    "HRT", "hormone therapy", "estrogen", "progesterone",
    "mood swings", "anxiety", "depression", "brain fog",
    "sleep issues", "exercise", "diet", "self-care"
]


# We will also loop through these sort methods to get more unique posts
sort_methods = ["relevance", "top", "new"]

# --- Keyword Set 1: Products & Stigma ---
keywords = ["menopause", "perimenopause", "hot flashes", "night sweats", "HRT", "hormone therapy", "estrogen", "progesterone", 
                     "mood swings", "anxiety", "depression", "brain fog", "sleep issues", "exercise", "diet", "self-care"]

# How many posts to scrape *per query*.
# 1000 is the max limit from Reddit's API.
POST_LIMIT_PER_QUERY = 10
POST_LIMIT_PER_QUERY = 1000

# --- NEW FUNCTION TO BUILD BETTER QUERIES ---
def create_query(keywords):
    """
    Creates a Reddit search query string.
    Puts quotes around multi-word phrases, but not single words.
    e.g., "menstrual cup" OR tampons OR TSS
    """
    query_parts = []
    for k in keywords:
        if " " in k:
            query_parts.append(f'"{k}"') # Add quotes if it's a multi-word phrase
        else:
            query_parts.append(k) # No quotes for single words
    return ' OR '.join(query_parts)

# --- Build the new, more effective queries ---
query = create_query(keywords)
output_filename = "data/raw/raw_reddit_data.csv"
processed_post_ids_products = set()


# --- 3. HELPER FUNCTION TO PROCESS POSTS ---
def process_submissions(submissions, writer, processed_post_ids):
# ... (This function is unchanged from before) ...
    """
    Takes a list of PRAW submissions, a CSV writer, and a set of processed IDs.
    Loops through submissions, gets all comments, and writes to the CSV.
    Returns the number of new posts and comments saved.
    """
    comments_saved_in_batch = 0
    posts_saved_in_batch = 0

    # We must loop through the generator to see if it's empty
    posts_found_in_query = 0

    for post in submissions:
        posts_found_in_query += 1
        # Check if we already processed this post
        if post.id in processed_post_ids:
            print(f"Skipping duplicate post: {post.id}")
            continue

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

    if posts_found_in_query == 0:
        print("No results found for this query.")

    return posts_saved_in_batch, comments_saved_in_batch


# --- 4. MAIN SCRAPING LOGIC ---
# ... (This section is unchanged from before, but will now use the new queries) ...

print(f"Connecting to Reddit...")
auth_success = False # <--- I've added a flag
# Test authentication
try:
    print(f"Logged in as: {reddit.user.me()}")
    if reddit.user.me() is None:
        print("Login failed, please check username/password. Running read-only.")
    else:
        auth_success = True # <--- Set the flag if login is good
except Exception as e:
    print(f"Authentication failed: {e}")
    print("Please check your client_id, client_secret, user_agent, and password.")
    print("\n>>> ERROR: DID YOU ADD YOUR REDDIT PASSWORD TO LINE 11? <<<\n")
    sys.exit() # <--- This will now reliably stop the script

if not auth_success:
    print("Authentication was not successful. Exiting script before starting scrapes.")
    sys.exit()

print(f"Starting advanced scrape...")
print(f"Sort Methods: {', '.join(sort_methods)}")
print(f"Post Limit per Query: {POST_LIMIT_PER_QUERY}")

# --- SCRAPE 1: PRODUCTS ---
print(f"\n\n{'='*20} STARTING SCRAPE {'='*20}")
print(f"Subreddits: {', '.join(subreddit_list)}")
print(f"Keywords: {query[:150]}...")
print(f"Output file: {output_filename}")

total_comments_saved_products = 0
total_unique_posts_saved_products = 0

with open(output_filename, 'w', newline='', encoding='utf-8') as f_obj:
    f_writer = csv.writer(f_obj)
    f_writer.writerow([
        "subreddit", "post_id", "post_created_utc", "post_title", "post_body",
        "post_score", "post_url", "comment_id", "comment_body", "comment_score",
        "comment_created_utc"
    ])

    for subreddit_name in subreddit_list:
        for sort_method in sort_methods:
            print(f"\n--- Querying r/{subreddit_name} (Sort: {sort_method}) ---")
            try:
                submissions = reddit.subreddit(subreddit_name).search(
                    query,
                    sort=sort_method,
                    time_filter="all",
                    limit=POST_LIMIT_PER_QUERY
                )

                new_posts, new_comments = process_submissions(
                    submissions,
                    f_writer,
                    processed_post_ids_products
                )

                total_unique_posts_saved_products += new_posts
                total_comments_saved_products += new_comments

            except Exception as e:
                print(f"!! FAILED to scrape r/{subreddit_name} with sort '{sort_method}': {e}")
                print("Moving to next query...")
                time.sleep(5) # Pause for 5 seconds before next query
                continue

print(f"\n--- SCRAPE COMPLETE ---")
print(f"Total Unique Posts: {total_unique_posts_saved_products}")
print(f"Total Comments: {total_comments_saved_products}")
print(f"Product data saved to: {output_filename}")


# print(f"\n\n{'='*20} OVERALL SCRAPE COMPLETE {'='*20}")