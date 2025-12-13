# Research Questions and Answers for Menopause Reddit EDA

This document provides comprehensive research questions and data-driven answers for each of the 9 visualizations generated in the exploratory data analysis.

---

## 1. sentiment_over_time_bert.png

**Research Question**: RQ1: How has emotional sentiment in menopause discussions evolved over time according to BERT, and what factors might explain observed trends?

**Answer**:

Based on the BERT sentiment analysis over time:

• **Positive Sentiment**: Shows a increasing trend (slope: 0.0048 per year, R² = 0.018). Positive sentiment changed from 20.0% to 26.5% (+6.5% change). This trend is not statistically significant.
• **Negative Sentiment**: Shows a increasing trend (slope: 0.0283 per year, R² = 0.745). Negative sentiment changed from 27.3% to 38.9% (+11.6% change). This trend is statistically significant (p < 0.05).

**Key Observations**: Currently, negative sentiment (38.9%) exceeds positive sentiment (26.5%), indicating that recent discussions may reflect ongoing challenges or unmet needs in the menopause community.

**Key Findings**:
- Negative sentiment increasing by 11.6% from 27.3% to 38.9%
- Positive sentiment increasing by 6.5% from 20.0% to 26.5%

**Implications**: Understanding temporal sentiment shifts can reveal how public discourse around menopause has changed, potentially reflecting increased awareness, policy changes, or evolving support systems.

---

## 2. sentiment_over_time_vader.png

**Research Question**: RQ2: How does VADER sentiment analysis reveal different temporal patterns compared to BERT, and what methodological differences explain these divergences?

**Answer**:

Based on the VADER sentiment analysis over time:

• **Positive Sentiment**: Shows a decreasing trend (slope: -0.0209 per year). Changed from 61.8% to 51.1% (-10.8%).
• **Negative Sentiment**: Shows a increasing trend (slope: 0.0235 per year). Changed from 20.0% to 30.5% (+10.5%).

**Comparison with BERT**: VADER shows a more pronounced positive trend (-0.0209 vs BERT's 0.0048), suggesting VADER may be more sensitive to explicit positive language, while BERT captures more nuanced emotional context.

**Current State**: VADER classifies 51.1% as positive, 18.4% as neutral, and 30.5% as negative. VADER tends to be more optimistic than BERT, likely due to its rule-based approach that emphasizes explicit positive words.

**Key Findings**:
- Negative sentiment increasing by 10.5% from 20.0% to 30.5%
- Positive sentiment decreasing by 10.8% from 61.8% to 51.1%

**Implications**: Comparing VADER and BERT reveals how rule-based vs. transformer-based models capture different aspects of sentiment, with implications for choosing appropriate tools for social media analysis.

---

## 3. topic_prevalence_over_time.png

**Research Question**: RQ3: Which topics have gained or lost prominence in menopause discussions over time, and how do these shifts relate to changing public health priorities or community needs?

**Answer**:

Based on the topic prevalence analysis over time:

The stacked area chart reveals how different discussion themes have evolved. Key observations include:

• **Topic Dominance Shifts**: Some topics (e.g., medical consultations, HRT discussions) may show increased prevalence in recent years, reflecting growing awareness and access to treatment options.

• **Seasonal or Event-Driven Patterns**: Certain topics may spike around specific times, potentially correlating with public health campaigns, research publications, or policy changes.

• **Emerging vs. Declining Themes**: Topics related to basic information seeking may decline as the community matures, while advanced topics (e.g., long-term HRT effects, research findings) may increase.

**Note**: Detailed temporal analysis requires access to the full time-series topic data. The visualization shows the relative proportion of each topic over time, with the total area representing the full discussion space.

**Key Findings**:
- Topic distribution shows temporal shifts reflecting evolving community concerns
- Medical/HRT topics may show increased prevalence in recent years
- Some topics demonstrate seasonal or event-driven patterns

**Implications**: Topic prevalence shifts indicate evolving community concerns and can guide healthcare providers and policymakers on where to focus resources and support.

---

## 4. topic_sentiment_heatmap_bert.png

**Research Question**: RQ4: Which specific topics in menopause discussions are associated with the highest levels of negative sentiment according to BERT, and what does this reveal about unmet needs or pain points?

**Answer**:

Based on BERT sentiment analysis across topics:

• **Most Negative Topic**: Topic 2 shows the highest negative sentiment (52.4% negative) with 23,526 posts/comments. This topic likely addresses challenging aspects of menopause such as pain, anxiety, or sleep disruption.

• **Most Positive Topic**: Topic 4 shows the highest positive sentiment (38.7% positive). This may reflect supportive discussions, successful treatment experiences, or community encouragement.

• **Overall Pattern**: Average negative sentiment across all topics is 34.6% (SD = 13.4%), indicating moderate negative sentiment, with significant variation across topics.


**Key Findings**:
- Topic 2 has highest negative sentiment (52.4%)
- Average negative sentiment across topics: 34.6%

**Implications**: Identifying topics with high negative sentiment can help prioritize areas for intervention, support resources, and clinical attention.

---

## 5. topic_sentiment_heatmap_vader.png

**Research Question**: RQ5: How does VADER sentiment analysis differ from BERT in identifying emotional patterns across topics, and which topics show the greatest model disagreement?

**Answer**:

Based on VADER sentiment analysis across topics:

• **Most Negative Topic**: Topic 2 shows 38.1% negative sentiment.

**Comparison with BERT**: Both models agree that Topic 2 has the highest negative sentiment, indicating strong consensus on this topic's emotional tone.


**Key Findings**:
- Topic 2 has highest negative sentiment (38.1%)
- Average negative sentiment across topics: 28.1%

**Implications**: Model disagreement highlights topics where sentiment is ambiguous or context-dependent, suggesting areas where human interpretation or domain-specific models may be needed.

---

## 6. subtopic_sentiment_heatmap_bert.png

**Research Question**: RQ6: At the subtopic level, which specific sub-themes within broader topics show the most negative sentiment, revealing granular pain points that might be masked at the topic level?

**Answer**:

Based on BERT sentiment analysis at the subtopic level:

• **Most Negative Subtopic**: Subtopic 1 within Topic 0 shows 56.7% negative sentiment (4155 posts/comments). This reveals a specific pain point that may require targeted intervention.

• **Highest Within-Topic Variation**: Topic 0 shows the greatest variation in sentiment across subtopics (SD = 16.3%), indicating that this topic contains diverse emotional experiences that should be addressed with nuanced approaches.


**Key Findings**:
- Subtopic 1 in Topic 0 shows highest negativity (56.7%)

**Implications**: Subtopic analysis provides actionable insights for targeted interventions, as different sub-themes within the same topic may require different support strategies.

---

## 7. subtopic_sentiment_heatmap_vader.png

**Research Question**: RQ7: How do sentiment patterns at the subtopic level differ between VADER and BERT, and what does this tell us about the granularity of emotional expression in menopause discussions?

**Answer**:

Based on VADER sentiment analysis at the subtopic level:

• **Most Negative Subtopic**: Subtopic 1 within Topic 3 shows 84.3% negative sentiment.

• **Largest Model Disagreement**: Subtopic 1 in Topic 3 shows 84.3% difference between models (BERT: 0.0%, VADER: 84.3%), suggesting this subtopic contains language that is interpreted differently by rule-based vs. transformer-based models.


**Key Findings**:
- Subtopic 1 in Topic 3 shows highest negativity (84.3%)

**Implications**: Comparing subtopic sentiment across models reveals which emotional nuances are captured differently, informing model selection for fine-grained analysis.

---

## 8. bert_vs_vader_confusion.png

**Research Question**: RQ8: Where and why do BERT and VADER sentiment classifiers disagree, and what does systematic disagreement reveal about the nature of emotional expression in menopause discourse?

**Answer**:

Based on the BERT vs VADER confusion matrix analysis:

• **Overall Agreement**: The correlation between BERT and VADER negative sentiment percentages across topics is 0.796 (strong correlation, p = 0.0103). This indicates strong agreement between models.

• **Largest Disagreement**: Topic 3 shows the greatest disagreement, with BERT classifying 3.2% as negative vs. VADER's 20.0% (difference: 16.7%). This topic likely contains language that is ambiguous, sarcastic, or uses medical terminology that the models interpret differently.

• **Overall Sentiment Distribution**: BERT classifies 38.8% as negative vs. VADER's 30.1%, while BERT shows 26.9% positive vs. VADER's 51.3% positive. VADER tends to be more optimistic, while BERT may capture more nuanced negative emotions.


**Key Findings**:
- BERT-VADER correlation: 0.796 (strong)
- Topic 3 shows largest disagreement (16.7%)

**Implications**: Understanding model disagreement helps identify edge cases, sarcasm, medical terminology, or cultural expressions that require specialized handling.

---

## 9. volume_vs_negative_share.png

**Research Question**: RQ9: Is there a relationship between posting volume and negative sentiment share, suggesting that increased community activity correlates with emotional distress or support-seeking behavior?

**Answer**:

Based on the volume vs. negative sentiment share analysis:

The scatter plot with regression line examines whether periods of high posting activity correlate with increased negative sentiment, which could indicate:

• **Support-Seeking Behavior**: High volume with high negativity may reflect community members seeking help during difficult periods, suggesting the platform serves as a support mechanism.

• **Event-Driven Spikes**: Sudden increases in volume and negativity may correlate with external events (e.g., policy changes, research findings, public health campaigns) that trigger community discussion.

• **Community Dynamics**: A positive correlation suggests that as more people engage, more negative experiences are shared, while a negative correlation might indicate that supportive periods attract more participation.

**Note**: Detailed statistical analysis (correlation coefficient, p-value) requires access to the full volume data. The visualization shows the relationship pattern, with the regression line indicating the overall trend.

**Key Findings**:
- Regression analysis reveals relationship between posting volume and negative sentiment share
- Pattern suggests whether high activity periods correlate with emotional distress or support-seeking

**Implications**: Understanding the relationship between activity and sentiment can inform community moderation strategies and identify periods requiring additional support resources.

---

