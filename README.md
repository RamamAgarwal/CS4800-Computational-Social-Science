# Exploiting Computational Techniques to Understand Menopause on Reddit

---

## Abstract

This project presents a computational analysis of menopause-related conversations on Reddit, focusing on themes, temporal sentiment dynamics, and community engagement patterns. We collected posts and comments from menopause-focused subreddits and applied preprocessing (PII removal, tokenization, lemmatization), exploratory data analysis (EDA), topic modeling using Latent Dirichlet Allocation (LDA), and sentiment analysis comparing VADER (rule-based) and BERT (transformer-based) models. We answer three main research questions: (1) How topic prevalence shifts over time, (2) How post sentiment relates to comment sentiment, and (3) Which topics/subtopics exhibit the highest negative sentiment and therefore highest intervention priority. Our analysis reveals significant temporal shifts in discussion topics, emotional contagion patterns in post-comment interactions, and identifies specific high-risk areas requiring targeted support.

**Keywords**: menopause, perimenopause, topic modeling, LDA, VADER, BERT, Reddit, sentiment analysis, temporal trends

---

## I. Introduction

### Objectives

1. Collect and preprocess Reddit posts and comments from menopause-focused communities.
2. Perform exploratory data analysis (EDA) to understand data characteristics and patterns.
3. Identify dominant themes using LDA topic modeling and analyze topic prevalence over time.
4. Apply sentiment analysis using both VADER and BERT models for comparative analysis.
5. Answer three key research questions addressing temporal topic shifts, post-comment sentiment relationships, and high-risk topic identification.
6. Measure engagement patterns (comments, upvotes) and posting activity across users and time.

---

## II. Related Work

This paper extends prior social-media health analyses by explicitly combining temporal topic prevalence analysis (RQ1), post–comment sentiment correlation analysis (RQ2), and targeted high-risk topic identification (RQ3) with practical EDA and model comparison steps. Our work builds on methodologies from Goel et al. (2023) on endometriosis social media analysis and Dhankar & Katz (2023) on tracking mental health through Reddit, adapting these approaches specifically for menopause discourse analysis.

---

## III. Data and Methods

### A. Data Sources & Collection

**Reddit Subreddits**: r/Menopause, r/Perimenopause, r/MenopauseSupport, r/EarlyMenopause, r/Hormones, r/WomensHealth

**Keyword Filter**: "menopause", "perimenopause", "hot flashes", "HRT", "estrogen", "progesterone", "brain fog", and related terms

**Dataset Characteristics**:
- **Total Comments Analyzed**: 167,150
- **Unique Posts**: 1,559
- **Time Period**: 2020-2025
- **Data Collection**: Posts and associated comments were scraped using PRAW (Python Reddit API Wrapper), with all personal identifiers removed for anonymization

### B. Preprocessing Pipeline

1. **Text Cleaning**:
   - Removal of Reddit usernames (u/username patterns)
   - Email and phone number removal
   - URL removal
   - Emoji removal
   - Normalization of whitespace

2. **Text Processing**:
   - Tokenization using spaCy
   - Stopword removal (English stopwords + domain-specific stopwords)
   - Lemmatization using spaCy's en_core_web_sm model
   - Short-text filtering (removed empty or very short texts)

3. **Feature Extraction**:
   - Created unified text fields (post_text_raw, comment_text_raw, text_raw)
   - Generated cleaned versions (post_text_clean, comment_text_clean, text_clean)
   - Generated lemmatized versions (post_text_lem, comment_text_lem, text_lem)

### C. Exploratory Data Analysis (EDA)

We performed comprehensive EDA including:

1. **Temporal Analysis**:
   - Sentiment trends over years (2020-2025) for both VADER and BERT
   - Topic prevalence shifts over time (monthly and yearly)

2. **Topic-Sentiment Analysis**:
   - Sentiment distribution across topics (BERT and VADER)
   - Sentiment distribution across subtopics
   - Identification of high-negative-sentiment topics/subtopics

3. **User Activity Analysis**:
   - Posting activity histogram (posts per user)
   - Top active users identification
   - Topic distribution for active users

4. **Post-Comment Relationship Analysis**:
   - Correlation between post sentiment and comment sentiment
   - Sentiment alignment matrices

### D. Topic Modeling: LDA

**Method**: Latent Dirichlet Allocation (LDA) using Gensim

**Configuration**:
- Explored K ∈ [3, 10] topics
- Selected optimal K using coherence score maximization (c_v metric)
- Final model: 9 topics (Topic IDs: 0-8)
- Bigram and trigram phrase modeling for improved topic quality
- Dictionary filtering: no_below=5, no_above=0.5
- Training: 12 passes, 10 workers, random_state=42

**Topic Interpretation**: Topics were interpreted based on top keywords and manual review. The model identified themes including hormonal adjustments, medical consultations, sleep issues, anxiety/pain, research discussions, intimacy concerns, and life reflections.

**Subtopic Modeling**: For each main topic, we performed additional LDA modeling on comment-level text to identify subtopics (2-8 subtopics per main topic), enabling granular analysis of discussion themes.

### E. Sentiment Analysis & Model Comparison

**Primary Models**:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
   - Rule-based sentiment analyzer optimized for social media
   - Returns compound score and individual positive/neutral/negative scores
   - Thresholds: ≥0.05 (positive), ≤-0.05 (negative), else (neutral)

2. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - Model: cardiffnlp/twitter-roberta-base-sentiment
   - Transformer-based classifier fine-tuned on Twitter data
   - Returns positive/neutral/negative labels with confidence scores
   - Handles context and nuance better than rule-based methods

**Comparison Rationale**: VADER provides fast, interpretable results optimized for social media language, while BERT captures more nuanced emotional context. Comparing both models validates findings and identifies edge cases where models disagree.

### F. Research Questions

**RQ1**: Which topics have gained or lost prominence in menopause discussions over time, and how do these shifts relate to changing public health priorities or community needs?

**RQ2**: How does the sentiment of a post affect the sentiment of its comments?

**RQ3**: Which topics and subtopics show the highest negative sentiment, and what does this reveal about areas requiring urgent intervention or support?

---

## IV. Results

### A. Dataset Overview

- **Total Comments**: 167,150
- **Unique Posts**: 1,559
- **Average Comments per Post**: ~107
- **Time Span**: 2020-2025 (6 years)
- **Overall Sentiment Distribution (BERT)**: 26.9% positive, 34.3% neutral, 38.8% negative
- **Overall Sentiment Distribution (VADER)**: 51.3% positive, 18.6% neutral, 30.1% negative

**Note**: VADER shows more optimistic classification than BERT, consistent with its rule-based approach that emphasizes explicit positive words.

### B. RQ1 — Topic Prevalence Over Time

**Question**: Which topics have gained or lost prominence in menopause discussions over time?

**Findings**:

**Top Gaining Topics** (2020–2025):
- **Topic 0**: +10.5% increase in discussion share
- **Topic 6**: +10.4% increase in discussion share
- **Topic 4**: +7.2% increase in discussion share

**Top Losing Topics** (2020–2025):
- **Topic 7**: -26.5% decrease in discussion share (largest decline)
- **Topic 1**: -6.1% decrease in discussion share
- **Topic 2**: -4.1% decrease in discussion share

**Interpretation**: 
- Growth patterns suggest shifting community focus toward topics reflecting increased awareness, education, and emerging concerns
- Declining topics may indicate information saturation, resolved issues, or migration of discussions to other platforms
- The substantial decline in Topic 7 (-26.5%) suggests either successful intervention or topic normalization

**Supporting Visualizations**:
- `rq1_topic_prevalence_timeline.png` - Monthly stacked area chart showing topic proportions over time
- `rq1_topic_gains_losses.png` - Bar charts highlighting top gaining and losing topics
- `topic_prevalence_over_time.png` - Comprehensive temporal view of all topics

### C. RQ2 — Post–Comment Sentiment Relationship

**Question**: How does the sentiment of a post affect the sentiment of its comments?

**Findings**:

**BERT Analysis**:
- **Correlation coefficient**: r = 0.426 (moderate positive correlation)
- **Statistical significance**: p < 0.0001 (highly significant)

**VADER Analysis**:
- **Correlation coefficient**: r = 0.287 (weak to moderate positive correlation)
- **Statistical significance**: p < 0.0001 (highly significant)

**Interpretation**:
- There is a moderate positive correlation between post sentiment and average comment sentiment
- Posts with more positive sentiment tend to receive comments with more positive sentiment (and vice versa)
- This reveals **emotional contagion** in the Reddit community: the emotional tone of a post influences the emotional tone of responses
- Positive posts attract supportive, positive comments, while negative posts may receive empathetic or problem-solving responses
- The community shows supportive response patterns rather than purely reactive behavior

**Practical Implications**:
- Moderators or community managers can influence discussion tone through post framing
- Supportive, positive posts may create more constructive discussion environments
- Negative posts, while receiving support, may benefit from early positive intervention

**Supporting Visualizations**:
- `rq2_post_comment_bert_scatter.png` - Scatter plot with regression line (BERT)
- `rq2_post_comment_vader_scatter.png` - Scatter plot with regression line (VADER)
- `rq2_sentiment_alignment_matrix.png` - Heatmap showing sentiment category alignment

### D. RQ3 — High-Risk Topics and Subtopics

**Question**: Which topics and subtopics show the highest negative sentiment, and what does this reveal about areas requiring urgent intervention or support?

**Findings**:

**Top 5 Most Negative Topics (BERT)**:
1. **Topic 2**: 52.4% negative sentiment (23,526 posts/comments) - **Highest priority**
2. **Topic 6**: 41.1% negative sentiment (16,819 posts/comments)
3. **Topic 7**: 40.8% negative sentiment (33,757 posts/comments) - **Largest volume**
4. **Topic 1**: 40.7% negative sentiment (22,824 posts/comments)
5. **Topic 4**: 34.7% negative sentiment (17,489 posts/comments)

**Top 5 Most Negative Subtopics (BERT)**:
1. **Topic 0 - Subtopic 1**: 56.7% negative (4,155 posts/comments) - **Highest negativity**
2. **Topic 2 - Subtopic 1**: 56.3% negative (3,788 posts/comments)
3. **Topic 2 - Subtopic 2**: 56.0% negative (2,911 posts/comments)
4. **Topic 2 - Subtopic 6**: 54.7% negative (3,441 posts/comments)
5. **Topic 2 - Subtopic 0**: 53.0% negative (3,269 posts/comments)

**Context**: The average negative sentiment across all topics is 34.6%. Topics/subtopics significantly above this average (especially those exceeding 50%) represent high-priority areas requiring urgent intervention.

**Key Insights**:
- **Topic 2** stands out with over 52% negative sentiment, making it the highest priority for intervention
- **Topic 7**, while having 40.8% negative sentiment, has the largest volume (33,757 posts), indicating widespread concern
- **Subtopic analysis** reveals specific pain points that may be masked at the topic level
- **Topic 0 - Subtopic 1** shows 56.7% negativity, the highest of all subtopics
- **Topic 2** contains multiple high-negativity subtopics (1, 2, 6, 0), indicating it's a particularly challenging topic area

**Model Validation**: BERT and VADER show agreement on the most problematic topics (both identify Topic 2 as highly negative), validating the findings. However, BERT shows higher negative percentages overall, likely due to better capture of nuanced negative emotions.

**Supporting Visualizations**:
- `rq3_top_negative_topics_bert.png` - Bar chart of top 5 most negative topics
- `rq3_top_negative_subtopics_bert.png` - Bar chart of top 8 most negative subtopics
- `rq3_model_comparison_heatmap.png` - BERT vs VADER comparison for validation
- `topic_sentiment_heatmap_bert.png` & `subtopic_sentiment_heatmap_bert.png` - Comprehensive sentiment distribution matrices

### E. Additional EDA Findings

**Posting Activity Distribution**:
- The posting distribution is heavily right-skewed
- Majority of users post once or very few times
- A small set of users are highly active (super-posters)
- This pattern is typical of online communities

**Sentiment Trends Over Time**:
- **BERT**: Negative sentiment increased from 27.3% (2020) to 38.9% (2025), a statistically significant trend (R² = 0.745, p < 0.05)
- **VADER**: Negative sentiment increased from 20.0% (2020) to 30.5% (2025)
- Both models show increasing negative sentiment over the 5-year period, potentially reflecting growing awareness of challenges or unmet needs

**Word Clouds**: Topic-specific word clouds were generated for each of the 9 main topics, showing discriminative keywords that help interpret topic themes (available in `outputs/figures/`).

---

## V. Discussion

### A. Temporal & Public-Health Relevance

The temporal analysis documents clear topic shifts, with some topics gaining over 10% share while others decline substantially (Topic 7: -26.5%). These shifts indicate evolving community concerns and potentially changing access to treatment, awareness campaigns, or information saturation. The increasing negative sentiment trend (2020-2025) may reflect:
- Growing awareness of menopause challenges
- Increased willingness to discuss difficult experiences
- Potential unmet needs requiring attention

These patterns should inform the timing and content of public-health messaging and resource allocation.

### B. Sentiment Dynamics & Community Behavior

The moderate positive correlation between post sentiment and comment sentiment (r = 0.426 for BERT, r = 0.287 for VADER) demonstrates **emotional contagion** in the Reddit community. This finding has practical implications:
- **For Community Moderators**: Encouraging positive framing in posts can foster supportive discussions
- **For Healthcare Providers**: Understanding emotional dynamics can inform communication strategies
- **For Researchers**: This pattern suggests the community serves as both a support mechanism and an emotional echo chamber

### C. Model Comparison & Validation

**VADER vs BERT Differences**:
- VADER shows more optimistic classification (51.3% positive vs BERT's 26.9%)
- BERT captures more nuanced negative emotions (38.8% negative vs VADER's 30.1%)
- Both models agree on directional relationships (e.g., Topic 2 as most negative)
- BERT shows stronger correlation for post-comment sentiment relationship (r = 0.426 vs 0.287)

**Implications**: 
- VADER is fast and interpretable but may miss domain-specific nuances
- BERT provides more accurate sentiment detection for clinical/health contexts
- Using both models provides validation and identifies edge cases

### D. High-Risk Topic Identification & Intervention Priorities

**Immediate Action Required**:
- **Topic 2** (52.4% negative, 23,526 posts) requires urgent support resources
- **Topic 0 - Subtopic 1** (56.7% negative) represents the highest-negativity area
- **Topic 7** (40.8% negative, 33,757 posts) needs attention due to large volume

**Targeted Support Strategies**:
- Subtopic-level analysis enables precise intervention strategies
- High-volume topics (Topic 7) may benefit from automated resource linking
- High-negativity topics (Topic 2) may require moderated support groups or clinical outreach

### E. Limitations

1. **Cross-sectional Analysis**: Our temporal analysis is observational; we cannot establish causal relationships between topic shifts and external events.

2. **Reddit Demographics**: Reddit user demographics may not represent the broader menopause population, limiting generalizability.

3. **Topic Interpretation**: LDA topics require manual interpretation; different researchers might interpret topics differently.

4. **Sentiment Model Limitations**: 
   - VADER may miss clinical terminology and nuanced expressions
   - BERT, while more accurate, requires computational resources
   - Neither model was fine-tuned on menopause-specific data

5. **Data Collection**: 
   - Reddit's API limitations may have affected data completeness
   - Keyword-based filtering may have missed relevant discussions
   - Temporal coverage (2020-2025) may not capture longer-term trends

6. **Anonymization**: While PII was removed, some contextual information may have been lost in preprocessing.

---

## VI. Future Work

1. **Model Fine-tuning**: Fine-tune BERT on manually labeled menopause posts to improve domain-specific sentiment detection.

2. **Longitudinal Causal Analysis**: Relate topical shifts to specific policy changes, research publications, or public health campaigns to establish causal relationships.

3. **Multi-platform Analysis**: Extend analysis to Twitter, Facebook groups, and other platforms to capture broader discourse.

4. **Real-time Dashboard**: Develop a monitoring dashboard for live tracking of topic prevalence, sentiment drift, and high-engagement posts.

5. **Clinical Integration**: Partner with healthcare providers to validate findings and develop evidence-based intervention strategies.

6. **Emotion Classification**: Extend beyond sentiment (positive/negative) to specific emotions (anxiety, frustration, hope, relief) for more nuanced analysis.

7. **User Network Analysis**: Analyze comment threads and user interactions to understand support network structures.

8. **Comparative Analysis**: Compare menopause discourse with other health conditions to identify unique patterns and needs.

---

## VII. Conclusion

Combining LDA topic modeling, temporal analyses, and comparative sentiment modeling (VADER and BERT) yields actionable insights for clinical communication and community support design. Our three research questions reveal:

1. **Temporal Evolution**: Significant topic shifts (some gaining >10%, others declining >25%) indicate evolving community needs and priorities.

2. **Community Dynamics**: Moderate positive correlation between post and comment sentiment (r = 0.426) demonstrates emotional contagion, informing moderation and support strategies.

3. **Intervention Priorities**: Topic 2 (52.4% negative) and Topic 0-Subtopic 1 (56.7% negative) represent high-priority areas requiring urgent support resources.

These findings provide evidence-based guidance for:
- **Healthcare Providers**: Focus clinical attention on high-negativity topics
- **Community Moderators**: Leverage sentiment dynamics to foster supportive discussions
- **Researchers**: Monitor topic shifts to evaluate intervention effectiveness
- **Policymakers**: Allocate resources to high-volume, high-negativity areas

The computational approach demonstrated here can be adapted for other health conditions and social media platforms, providing scalable methods for understanding patient experiences and community needs.

---

## Acknowledgments

We thank our supervisors VijayaChitra Modhukur and Rajesh Sharma for their guidance. The project followed ethical scraping and anonymization guidelines, with all personal identifiers removed before analysis.

---

## References

1. Goel, R., Modhukur, V., Täär, K., Salumets, A., Sharma, R., & Peters, M. (2023). Users' concerns about endometriosis on social media: Sentiment analysis and topic modeling study. *Journal of Medical Internet Research*, 25, e45381. https://doi.org/10.2196/45381

2. Dhankar, A., & Katz, A. (2023). Tracking pregnant women's mental health through social media: An analysis of Reddit posts. *JAMIA Open*, 6(4), ooad094. https://doi.org/10.1093/jamiaopen/ooad094

3. Postill, G., Hussain-Shamsy, N., Dephoure, S., Wong, A., Shore, E. M., Cooper, J., ... & Bogler, T. (2025). Evaluation of a Canadian social media platform for communicating perinatal health information during a pandemic. *PLOS Digital Health*, 4(4), e0000802. https://doi.org/10.1371/journal.pdig.0000802

4. Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216-225.

5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186.

---
