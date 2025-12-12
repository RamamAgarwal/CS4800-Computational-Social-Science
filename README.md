
# Menopause Reddit Analysis

This project analyzes Reddit discussions about **menopause and perimenopause** to identify major topics and sentiment patterns using NLP techniques.

## Structure
- `data/` — raw and processed datasets  
- `scripts/` — Python scripts for scraping, preprocessing, modeling, and visualization  
- `notebooks/` — Jupyter notebooks for exploration and experimentation  
- `outputs/` — generated models, visualizations, and reports  

## Ethical note

All data is anonymized before analysis. No Reddit usernames or personal identifiers are stored.

# Understanding Menopause on Reddit Using NLP  
### *A Computational Analysis of Themes & Sentiments in Menopause-Related Discussions*

---

## Overview  
This project applies **Natural Language Processing (NLP)** methods to understand how people discuss **menopause and perimenopause** on Reddit. Using a pipeline involving **data scraping, preprocessing, topic modeling (LDA), and sentiment analysis (VADER)**, the study uncovers dominant themes, emotional patterns, and unmet support needs expressed by users across menopause-related communities.

This repository contains the code, report, figures, and documentation associated with the project.

---

## Key Objectives  
- Collect and preprocess Reddit posts from menopause-related subreddits  
- Identify high-level themes using **Latent Dirichlet Allocation (LDA)**  
- Analyze emotional tone using **VADER sentiment analysis**  
- Visualize topic–sentiment interactions  
- Draw insights applicable to healthcare communication, digital health tools, and support systems  

---

## Tech Stack  
- **Python 3.10+**  
- Libraries:
  - `praw` or `psaw` (Reddit scraping)
  - `pandas`, `numpy`
  - `spaCy`, `nltk`
  - `gensim` (LDA)
  - `matplotlib`, `seaborn`, `wordcloud`
  - `vaderSentiment`
  - `pyLDAvis`

---

## Repository Structure  
```
Menopause-Reddit-Analysis/
│
├── data/
│   ├── raw/                # Raw scraped Reddit posts
│   ├── processed/          # Cleaned + lemmatized corpus
│
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lda_topic_modeling.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_visualizations.ipynb
│
├── src/
│   ├── scraping.py
│   ├── cleaning.py
│   ├── lda_model.py
│   ├── sentiment.py
│   ├── utils.py
│
├── results/
│   ├── lda_topics.csv
│   ├── sentiment_scores.csv
│   ├── figures/
│       ├── coherence_plot.png
│       ├── topic_wordclouds/
│       ├── sentiment_distribution.png
│       ├── pyldavis_intertopic_map.html
│
├── report/
│   ├── IEEE_Report.pdf
│   ├── IEEE_Report.tex (optional)
│
├── README.md
└── requirements.txt
```
---

## Data Preprocessing Pipeline  
Key steps:

1. **PII Removal**  
2. **Noise Cleaning (emojis, URLs, punctuation)**  
3. **Tokenization & Stopword Removal**  
4. **Lemmatization (spaCy)**  
5. **Short-text Filtering**  
6. **Vectorization for LDA (Gensim Dictionary + BoW)**  

---

## Topic Modeling (LDA)  
- Explored **K = 5–10 topics**  
- Selected **K = 7** using coherence score maximization  
- Final topics:

| ID | Theme | Description |
|----|--------|-------------|
| 1 | Cycle & Hormonal Adjustments | Menstrual irregularity, HRT, hot flashes |
| 2 | Doctor Consultations & HRT | Medical advice, estrogen, progesterone |
| 3 | Life Reflections | Mood changes, self-perception, aging |
| 4 | Sleep & Hot Flash Issues | Night sweats, insomnia |
| 5 | Anxiety & Pain | Emotional distress, chronic pain |
| 6 | Medical Research | Breast cancer risk, studies |
| 7 | Intimacy & Relationships | Sexual health, communication |

---

## Sentiment Analysis (VADER)  
Posts classified into **positive, neutral, and negative** categories.

### Key insights:
- **Highest negative sentiment:** Anxiety & Pain  
- Sleep & Hot Flashes → predominantly negative  
- Doctor Consultations & Medical Research → more neutral/positive  
- Intimacy & Relationship posts → mixed sentiment  

---

## Visualizations  
- Coherence Score Plot  
- LDA Word Clouds  
- pyLDAvis Intertopic Map  
- Sentiment Distribution Charts  
- Topic–Sentiment Stacked Bars  

All visual outputs are located in the `results/figures/` folder.

---

## Key Insights  
- Reddit is a significant platform for menopause discussions  
- Users show **high emotional burden** (anxiety, pain, sleep issues)  
- Discussions about **HRT and clinicians** tend to be hopeful  
- Need for **better clinical communication and digital health tools**  
- Social media analysis provides **real-time, large-scale patient perspectives**  

---

## Future Improvements  
- Use BERT/RoBERTa for advanced sentiment & emotion detection  
- Multi-label emotion classification  
- Temporal trend analysis  
- Expand across platforms (Twitter, Facebook)  
- Develop a monitoring dashboard  

---

## License  
This project is intended for academic and research use.  
Please follow Reddit’s policies for handling scraped data.

---

## ⭐ If you find this useful  
Please consider giving the repository a **star** 🌟 or citing the report.

>>>>>>> 29fdd3ebc3e6bd90d6a781caa2f75be923629858
