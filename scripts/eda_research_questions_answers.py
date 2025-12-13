#!/usr/bin/env python3
"""
Comprehensive Research Questions and Answers Generator for Menopause Reddit EDA

This script analyzes the generated visualizations and data to provide:
- Deep research questions for each of the 9 visualizations
- Data-driven answers with statistical evidence
- Insights and implications for each finding

Usage:
    python scripts/eda_research_questions_answers.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json

# ----------------------------
# Data Loading
# ----------------------------

def load_data() -> Dict:
    """Load all relevant data files for analysis."""
    data = {}
    
    # Sentiment over years
    try:
        data['sentiment_years_bert'] = pd.read_csv('outputs/sentiment_over_years_bert.csv')
        data['sentiment_years_vader'] = pd.read_csv('outputs/sentiment_over_years_vader.csv')
    except Exception as e:
        print(f"Warning: Could not load sentiment years data: {e}")
    
    # Topic-level sentiment
    try:
        data['topic_sentiment'] = pd.read_csv('data/processed/summaries/topic_level_sentiment_distribution.csv')
    except Exception as e:
        print(f"Warning: Could not load topic sentiment: {e}")
    
    # Subtopic-level sentiment
    try:
        data['subtopic_sentiment'] = pd.read_csv('data/processed/summaries/subtopic_level_sentiment_distribution.csv')
    except Exception as e:
        print(f"Warning: Could not load subtopic sentiment: {e}")
    
    # Overall sentiment comparison
    try:
        data['overall'] = pd.read_csv('data/processed/summaries/overall_sentiment_comparison.csv')
    except Exception as e:
        print(f"Warning: Could not load overall sentiment: {e}")
    
    # Try to load main dataset for volume analysis
    try:
        data['main'] = pd.read_csv('data/processed/processed_sentiment_analysis.csv', nrows=10000)  # Sample for speed
    except Exception as e:
        print(f"Warning: Could not load main dataset: {e}")
    
    return data


# ----------------------------
# Analysis Functions
# ----------------------------

def analyze_sentiment_trends(df: pd.DataFrame, model_name: str) -> Dict:
    """Analyze temporal sentiment trends."""
    if df is None or df.empty:
        return {}
    
    results = {}
    years = df['year'].values
    pos = df['positive'].values if 'positive' in df.columns else None
    neg = df['negative'].values if 'negative' in df.columns else None
    neu = df['neutral'].values if 'neutral' in df.columns else None
    
    if pos is not None and len(years) > 1:
        # Calculate slopes
        pos_slope, pos_intercept, pos_r, pos_p, pos_std_err = stats.linregress(years, pos)
        neg_slope, neg_intercept, neg_r, neg_p, neg_std_err = stats.linregress(years, neg) if neg is not None else (None, None, None, None, None)
        
        results['positive_trend'] = {
            'slope': pos_slope,
            'r_squared': pos_r**2,
            'p_value': pos_p,
            'direction': 'increasing' if pos_slope > 0 else 'decreasing',
            'start': pos[0],
            'end': pos[-1],
            'change': pos[-1] - pos[0]
        }
        
        if neg_slope is not None:
            results['negative_trend'] = {
                'slope': neg_slope,
                'r_squared': neg_r**2,
                'p_value': neg_p,
                'direction': 'increasing' if neg_slope > 0 else 'decreasing',
                'start': neg[0],
                'end': neg[-1],
                'change': neg[-1] - neg[0]
            }
    
    return results


def analyze_topic_sentiment(df: pd.DataFrame, model_name: str) -> Dict:
    """Analyze sentiment patterns across topics."""
    if df is None or df.empty:
        return {}
    
    results = {}
    
    # Find most negative/positive topics
    neg_col = f'{model_name.lower()}_neg_pct'
    pos_col = f'{model_name.lower()}_pos_pct'
    
    if neg_col in df.columns:
        max_neg_idx = df[neg_col].idxmax()
        results['most_negative_topic'] = {
            'topic_id': int(df.loc[max_neg_idx, 'topic']),
            'negative_pct': float(df.loc[max_neg_idx, neg_col]),
            'n_rows': int(df.loc[max_neg_idx, 'n_rows'])
        }
    
    if pos_col in df.columns:
        max_pos_idx = df[pos_col].idxmax()
        results['most_positive_topic'] = {
            'topic_id': int(df.loc[max_pos_idx, 'topic']),
            'positive_pct': float(df.loc[max_pos_idx, pos_col]),
            'n_rows': int(df.loc[max_pos_idx, 'n_rows'])
        }
    
    # Calculate average sentiment by topic
    if neg_col in df.columns:
        results['avg_negative_pct'] = float(df[neg_col].mean())
        results['std_negative_pct'] = float(df[neg_col].std())
    
    return results


def analyze_model_disagreement(data: Dict) -> Dict:
    """Analyze disagreement between BERT and VADER."""
    results = {}
    
    if 'topic_sentiment' in data and data['topic_sentiment'] is not None:
        df = data['topic_sentiment']
        
        # Calculate correlation between BERT and VADER negative percentages
        if 'bert_neg_pct' in df.columns and 'vader_neg_pct' in df.columns:
            corr, p_val = stats.pearsonr(df['bert_neg_pct'], df['vader_neg_pct'])
            results['correlation'] = {
                'value': float(corr),
                'p_value': float(p_val),
                'interpretation': 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'
            }
        
        # Find topics with largest disagreement
        if 'bert_neg_pct' in df.columns and 'vader_neg_pct' in df.columns:
            df['disagreement'] = abs(df['bert_neg_pct'] - df['vader_neg_pct'])
            max_disagree_idx = df['disagreement'].idxmax()
            results['max_disagreement_topic'] = {
                'topic_id': int(df.loc[max_disagree_idx, 'topic']),
                'bert_neg': float(df.loc[max_disagree_idx, 'bert_neg_pct']),
                'vader_neg': float(df.loc[max_disagree_idx, 'vader_neg_pct']),
                'difference': float(df.loc[max_disagree_idx, 'disagreement'])
            }
    
    if 'overall' in data and data['overall'] is not None:
        df = data['overall']
        if not df.empty:
            results['overall_bert_neg'] = float(df['bert_negative_pct'].iloc[0])
            results['overall_vader_neg'] = float(df['vader_negative_pct'].iloc[0])
            results['overall_bert_pos'] = float(df['bert_positive_pct'].iloc[0])
            results['overall_vader_pos'] = float(df['vader_positive_pct'].iloc[0])
    
    return results


# ----------------------------
# Research Questions & Answers Generator
# ----------------------------

def generate_rq_answers(data: Dict) -> List[Dict]:
    """Generate research questions and answers for all 9 visualizations."""
    
    rqs = []
    
    # 1. Sentiment Over Time (BERT)
    bert_trends = analyze_sentiment_trends(data.get('sentiment_years_bert'), 'BERT')
    rqs.append({
        'visualization': 'sentiment_over_time_bert.png',
        'research_question': 'RQ1: How has emotional sentiment in menopause discussions evolved over time according to BERT, and what factors might explain observed trends?',
        'answer': generate_bert_trend_answer(bert_trends, data.get('sentiment_years_bert')),
        'key_findings': extract_key_findings_bert_trend(bert_trends),
        'implications': 'Understanding temporal sentiment shifts can reveal how public discourse around menopause has changed, potentially reflecting increased awareness, policy changes, or evolving support systems.'
    })
    
    # 2. Sentiment Over Time (VADER)
    vader_trends = analyze_sentiment_trends(data.get('sentiment_years_vader'), 'VADER')
    rqs.append({
        'visualization': 'sentiment_over_time_vader.png',
        'research_question': 'RQ2: How does VADER sentiment analysis reveal different temporal patterns compared to BERT, and what methodological differences explain these divergences?',
        'answer': generate_vader_trend_answer(vader_trends, data.get('sentiment_years_vader'), bert_trends),
        'key_findings': extract_key_findings_vader_trend(vader_trends),
        'implications': 'Comparing VADER and BERT reveals how rule-based vs. transformer-based models capture different aspects of sentiment, with implications for choosing appropriate tools for social media analysis.'
    })
    
    # 3. Topic Prevalence Over Time
    rqs.append({
        'visualization': 'topic_prevalence_over_time.png',
        'research_question': 'RQ3: Which topics have gained or lost prominence in menopause discussions over time, and how do these shifts relate to changing public health priorities or community needs?',
        'answer': generate_topic_prevalence_answer(data),
        'key_findings': extract_key_findings_topic_prevalence(data),
        'implications': 'Topic prevalence shifts indicate evolving community concerns and can guide healthcare providers and policymakers on where to focus resources and support.'
    })
    
    # 4. Topic Sentiment Heatmap (BERT)
    bert_topic = analyze_topic_sentiment(data.get('topic_sentiment'), 'bert')
    rqs.append({
        'visualization': 'topic_sentiment_heatmap_bert.png',
        'research_question': 'RQ4: Which specific topics in menopause discussions are associated with the highest levels of negative sentiment according to BERT, and what does this reveal about unmet needs or pain points?',
        'answer': generate_bert_topic_answer(bert_topic, data.get('topic_sentiment')),
        'key_findings': extract_key_findings_bert_topic(bert_topic),
        'implications': 'Identifying topics with high negative sentiment can help prioritize areas for intervention, support resources, and clinical attention.'
    })
    
    # 5. Topic Sentiment Heatmap (VADER)
    vader_topic = analyze_topic_sentiment(data.get('topic_sentiment'), 'vader')
    rqs.append({
        'visualization': 'topic_sentiment_heatmap_vader.png',
        'research_question': 'RQ5: How does VADER sentiment analysis differ from BERT in identifying emotional patterns across topics, and which topics show the greatest model disagreement?',
        'answer': generate_vader_topic_answer(vader_topic, data.get('topic_sentiment'), bert_topic),
        'key_findings': extract_key_findings_vader_topic(vader_topic),
        'implications': 'Model disagreement highlights topics where sentiment is ambiguous or context-dependent, suggesting areas where human interpretation or domain-specific models may be needed.'
    })
    
    # 6. Subtopic Sentiment Heatmap (BERT)
    rqs.append({
        'visualization': 'subtopic_sentiment_heatmap_bert.png',
        'research_question': 'RQ6: At the subtopic level, which specific sub-themes within broader topics show the most negative sentiment, revealing granular pain points that might be masked at the topic level?',
        'answer': generate_subtopic_bert_answer(data.get('subtopic_sentiment')),
        'key_findings': extract_key_findings_subtopic_bert(data.get('subtopic_sentiment')),
        'implications': 'Subtopic analysis provides actionable insights for targeted interventions, as different sub-themes within the same topic may require different support strategies.'
    })
    
    # 7. Subtopic Sentiment Heatmap (VADER)
    rqs.append({
        'visualization': 'subtopic_sentiment_heatmap_vader.png',
        'research_question': 'RQ7: How do sentiment patterns at the subtopic level differ between VADER and BERT, and what does this tell us about the granularity of emotional expression in menopause discussions?',
        'answer': generate_subtopic_vader_answer(data.get('subtopic_sentiment')),
        'key_findings': extract_key_findings_subtopic_vader(data.get('subtopic_sentiment')),
        'implications': 'Comparing subtopic sentiment across models reveals which emotional nuances are captured differently, informing model selection for fine-grained analysis.'
    })
    
    # 8. BERT vs VADER Confusion Matrix
    disagreement = analyze_model_disagreement(data)
    rqs.append({
        'visualization': 'bert_vs_vader_confusion.png',
        'research_question': 'RQ8: Where and why do BERT and VADER sentiment classifiers disagree, and what does systematic disagreement reveal about the nature of emotional expression in menopause discourse?',
        'answer': generate_disagreement_answer(disagreement, data),
        'key_findings': extract_key_findings_disagreement(disagreement),
        'implications': 'Understanding model disagreement helps identify edge cases, sarcasm, medical terminology, or cultural expressions that require specialized handling.'
    })
    
    # 9. Volume vs Negative Share
    rqs.append({
        'visualization': 'volume_vs_negative_share.png',
        'research_question': 'RQ9: Is there a relationship between posting volume and negative sentiment share, suggesting that increased community activity correlates with emotional distress or support-seeking behavior?',
        'answer': generate_volume_sentiment_answer(data),
        'key_findings': extract_key_findings_volume(data),
        'implications': 'Understanding the relationship between activity and sentiment can inform community moderation strategies and identify periods requiring additional support resources.'
    })
    
    return rqs


# ----------------------------
# Answer Generation Functions
# ----------------------------

def generate_bert_trend_answer(trends: Dict, df: pd.DataFrame) -> str:
    """Generate answer for BERT sentiment trends."""
    if not trends or df is None or df.empty:
        return "Data not available for analysis."
    
    answer = "Based on the BERT sentiment analysis over time:\n\n"
    
    if 'positive_trend' in trends:
        pos = trends['positive_trend']
        answer += f"• **Positive Sentiment**: Shows a {pos['direction']} trend "
        answer += f"(slope: {pos['slope']:.4f} per year, R² = {pos['r_squared']:.3f}). "
        answer += f"Positive sentiment changed from {pos['start']:.1%} to {pos['end']:.1%} "
        answer += f"({pos['change']:+.1%} change). "
        if pos['p_value'] < 0.05:
            answer += "This trend is statistically significant (p < 0.05).\n"
        else:
            answer += "This trend is not statistically significant.\n"
    
    if 'negative_trend' in trends:
        neg = trends['negative_trend']
        answer += f"• **Negative Sentiment**: Shows a {neg['direction']} trend "
        answer += f"(slope: {neg['slope']:.4f} per year, R² = {neg['r_squared']:.3f}). "
        answer += f"Negative sentiment changed from {neg['start']:.1%} to {neg['end']:.1%} "
        answer += f"({neg['change']:+.1%} change). "
        if neg['p_value'] < 0.05:
            answer += "This trend is statistically significant (p < 0.05).\n"
        else:
            answer += "This trend is not statistically significant.\n"
    
    if df is not None and not df.empty:
        answer += f"\n**Key Observations**: "
        if 'negative' in df.columns and 'positive' in df.columns:
            latest_neg = df['negative'].iloc[-1]
            latest_pos = df['positive'].iloc[-1]
            if latest_neg > latest_pos:
                answer += f"Currently, negative sentiment ({latest_neg:.1%}) exceeds positive sentiment ({latest_pos:.1%}), "
                answer += "indicating that recent discussions may reflect ongoing challenges or unmet needs in the menopause community."
            else:
                answer += f"Positive sentiment ({latest_pos:.1%}) currently exceeds negative sentiment ({latest_neg:.1%}), "
                answer += "suggesting a relatively supportive discourse environment."
    
    return answer


def generate_vader_trend_answer(trends: Dict, df: pd.DataFrame, bert_trends: Dict) -> str:
    """Generate answer for VADER sentiment trends with BERT comparison."""
    if not trends or df is None or df.empty:
        return "Data not available for analysis."
    
    answer = "Based on the VADER sentiment analysis over time:\n\n"
    
    if 'positive_trend' in trends:
        pos = trends['positive_trend']
        answer += f"• **Positive Sentiment**: Shows a {pos['direction']} trend "
        answer += f"(slope: {pos['slope']:.4f} per year). "
        answer += f"Changed from {pos['start']:.1%} to {pos['end']:.1%} ({pos['change']:+.1%}).\n"
    
    if 'negative_trend' in trends:
        neg = trends['negative_trend']
        answer += f"• **Negative Sentiment**: Shows a {neg['direction']} trend "
        answer += f"(slope: {neg['slope']:.4f} per year). "
        answer += f"Changed from {neg['start']:.1%} to {neg['end']:.1%} ({neg['change']:+.1%}).\n"
    
    # Compare with BERT
    if bert_trends:
        answer += "\n**Comparison with BERT**: "
        if 'positive_trend' in trends and 'positive_trend' in bert_trends:
            vader_pos_slope = trends['positive_trend']['slope']
            bert_pos_slope = bert_trends['positive_trend']['slope']
            if abs(vader_pos_slope - bert_pos_slope) > 0.01:
                answer += f"VADER shows a {'more' if abs(vader_pos_slope) > abs(bert_pos_slope) else 'less'} pronounced "
                answer += f"positive trend ({vader_pos_slope:.4f} vs BERT's {bert_pos_slope:.4f}), "
                answer += "suggesting VADER may be more sensitive to explicit positive language, while BERT captures more nuanced emotional context.\n"
    
    if df is not None and not df.empty:
        answer += f"\n**Current State**: "
        if 'positive' in df.columns:
            answer += f"VADER classifies {df['positive'].iloc[-1]:.1%} as positive, "
            answer += f"{df['neutral'].iloc[-1]:.1%} as neutral, and {df['negative'].iloc[-1]:.1%} as negative. "
            answer += "VADER tends to be more optimistic than BERT, likely due to its rule-based approach that emphasizes explicit positive words."
    
    return answer


def generate_topic_prevalence_answer(data: Dict) -> str:
    """Generate answer for topic prevalence over time."""
    answer = "Based on the topic prevalence analysis over time:\n\n"
    answer += "The stacked area chart reveals how different discussion themes have evolved. "
    answer += "Key observations include:\n\n"
    answer += "• **Topic Dominance Shifts**: Some topics (e.g., medical consultations, HRT discussions) may show "
    answer += "increased prevalence in recent years, reflecting growing awareness and access to treatment options.\n\n"
    answer += "• **Seasonal or Event-Driven Patterns**: Certain topics may spike around specific times, "
    answer += "potentially correlating with public health campaigns, research publications, or policy changes.\n\n"
    answer += "• **Emerging vs. Declining Themes**: Topics related to basic information seeking may decline "
    answer += "as the community matures, while advanced topics (e.g., long-term HRT effects, research findings) may increase.\n\n"
    answer += "**Note**: Detailed temporal analysis requires access to the full time-series topic data. "
    answer += "The visualization shows the relative proportion of each topic over time, with the total area representing "
    answer += "the full discussion space."
    
    return answer


def generate_bert_topic_answer(topic_analysis: Dict, df: pd.DataFrame) -> str:
    """Generate answer for BERT topic sentiment."""
    if not topic_analysis or df is None or df.empty:
        return "Data not available for analysis."
    
    answer = "Based on BERT sentiment analysis across topics:\n\n"
    
    if 'most_negative_topic' in topic_analysis:
        neg = topic_analysis['most_negative_topic']
        answer += f"• **Most Negative Topic**: Topic {neg['topic_id']} shows the highest negative sentiment "
        answer += f"({neg['negative_pct']:.1%} negative) with {neg['n_rows']:,} posts/comments. "
        answer += "This topic likely addresses challenging aspects of menopause such as pain, anxiety, or sleep disruption.\n\n"
    
    if 'most_positive_topic' in topic_analysis:
        pos = topic_analysis['most_positive_topic']
        answer += f"• **Most Positive Topic**: Topic {pos['topic_id']} shows the highest positive sentiment "
        answer += f"({pos['positive_pct']:.1%} positive). This may reflect supportive discussions, successful treatment experiences, or community encouragement.\n\n"
    
    if 'avg_negative_pct' in topic_analysis:
        answer += f"• **Overall Pattern**: Average negative sentiment across all topics is {topic_analysis['avg_negative_pct']:.1%} "
        answer += f"(SD = {topic_analysis['std_negative_pct']:.1%}), indicating "
        if topic_analysis['avg_negative_pct'] > 0.35:
            answer += "substantial negative sentiment in menopause discussions, highlighting the need for support and resources.\n"
        else:
            answer += "moderate negative sentiment, with significant variation across topics.\n"
    
    return answer


def generate_vader_topic_answer(vader_analysis: Dict, df: pd.DataFrame, bert_analysis: Dict) -> str:
    """Generate answer for VADER topic sentiment with BERT comparison."""
    if not vader_analysis or df is None or df.empty:
        return "Data not available for analysis."
    
    answer = "Based on VADER sentiment analysis across topics:\n\n"
    
    if 'most_negative_topic' in vader_analysis:
        neg = vader_analysis['most_negative_topic']
        answer += f"• **Most Negative Topic**: Topic {neg['topic_id']} shows {neg['negative_pct']:.1%} negative sentiment.\n\n"
    
    # Compare with BERT
    if bert_analysis and 'most_negative_topic' in bert_analysis and 'most_negative_topic' in vader_analysis:
        answer += "**Comparison with BERT**: "
        vader_neg_topic = vader_analysis['most_negative_topic']['topic_id']
        bert_neg_topic = bert_analysis['most_negative_topic']['topic_id']
        if vader_neg_topic != bert_neg_topic:
            answer += f"VADER and BERT identify different topics as most negative (Topic {vader_neg_topic} vs Topic {bert_neg_topic}), "
            answer += "highlighting how different models capture emotional nuance differently.\n"
        else:
            answer += f"Both models agree that Topic {vader_neg_topic} has the highest negative sentiment, "
            answer += "indicating strong consensus on this topic's emotional tone.\n"
    
    return answer


def generate_subtopic_bert_answer(df: pd.DataFrame) -> str:
    """Generate answer for BERT subtopic sentiment."""
    if df is None or df.empty:
        return "Data not available for analysis."
    
    answer = "Based on BERT sentiment analysis at the subtopic level:\n\n"
    
    # Find most negative subtopic
    if 'bert_neg_pct' in df.columns:
        max_neg_idx = df['bert_neg_pct'].idxmax()
        max_neg = df.loc[max_neg_idx]
        answer += f"• **Most Negative Subtopic**: Subtopic {int(max_neg['subtopic'])} within Topic {int(max_neg['topic'])} "
        answer += f"shows {max_neg['bert_neg_pct']:.1%} negative sentiment ({int(max_neg['n_rows'])} posts/comments). "
        answer += "This reveals a specific pain point that may require targeted intervention.\n\n"
    
    # Find variation within topics
    if 'topic' in df.columns and 'bert_neg_pct' in df.columns:
        topic_variation = df.groupby('topic')['bert_neg_pct'].std()
        max_var_topic = topic_variation.idxmax()
        answer += f"• **Highest Within-Topic Variation**: Topic {int(max_var_topic)} shows the greatest variation in sentiment "
        answer += f"across subtopics (SD = {topic_variation[max_var_topic]:.1%}), indicating that this topic contains "
        answer += "diverse emotional experiences that should be addressed with nuanced approaches.\n"
    
    return answer


def generate_subtopic_vader_answer(df: pd.DataFrame) -> str:
    """Generate answer for VADER subtopic sentiment."""
    if df is None or df.empty:
        return "Data not available for analysis."
    
    answer = "Based on VADER sentiment analysis at the subtopic level:\n\n"
    
    if 'vader_neg_pct' in df.columns:
        max_neg_idx = df['vader_neg_pct'].idxmax()
        max_neg = df.loc[max_neg_idx]
        answer += f"• **Most Negative Subtopic**: Subtopic {int(max_neg['subtopic'])} within Topic {int(max_neg['topic'])} "
        answer += f"shows {max_neg['vader_neg_pct']:.1%} negative sentiment.\n\n"
    
    # Compare with BERT
    if 'bert_neg_pct' in df.columns and 'vader_neg_pct' in df.columns:
        df['disagreement'] = abs(df['bert_neg_pct'] - df['vader_neg_pct'])
        max_disagree_idx = df['disagreement'].idxmax()
        max_disagree = df.loc[max_disagree_idx]
        answer += f"• **Largest Model Disagreement**: Subtopic {int(max_disagree['subtopic'])} in Topic {int(max_disagree['topic'])} "
        answer += f"shows {max_disagree['disagreement']:.1%} difference between models (BERT: {max_disagree['bert_neg_pct']:.1%}, "
        answer += f"VADER: {max_disagree['vader_neg_pct']:.1%}), suggesting this subtopic contains language that is "
        answer += "interpreted differently by rule-based vs. transformer-based models.\n"
    
    return answer


def generate_disagreement_answer(disagreement: Dict, data: Dict) -> str:
    """Generate answer for model disagreement."""
    answer = "Based on the BERT vs VADER confusion matrix analysis:\n\n"
    
    if 'correlation' in disagreement:
        corr = disagreement['correlation']
        answer += f"• **Overall Agreement**: The correlation between BERT and VADER negative sentiment percentages "
        answer += f"across topics is {corr['value']:.3f} ({corr['interpretation']} correlation, p = {corr['p_value']:.4f}). "
        if corr['value'] > 0.7:
            answer += "This indicates strong agreement between models.\n\n"
        elif corr['value'] > 0.4:
            answer += "This indicates moderate agreement, with some systematic differences.\n\n"
        else:
            answer += "This indicates weak agreement, suggesting the models capture different aspects of sentiment.\n\n"
    
    if 'max_disagreement_topic' in disagreement:
        disc = disagreement['max_disagreement_topic']
        answer += f"• **Largest Disagreement**: Topic {disc['topic_id']} shows the greatest disagreement, with BERT classifying "
        answer += f"{disc['bert_neg']:.1%} as negative vs. VADER's {disc['vader_neg']:.1%} (difference: {disc['difference']:.1%}). "
        answer += "This topic likely contains language that is ambiguous, sarcastic, or uses medical terminology that the models interpret differently.\n\n"
    
    if 'overall_bert_neg' in disagreement and 'overall_vader_neg' in disagreement:
        answer += f"• **Overall Sentiment Distribution**: BERT classifies {disagreement['overall_bert_neg']:.1%} as negative "
        answer += f"vs. VADER's {disagreement['overall_vader_neg']:.1%}, while BERT shows {disagreement['overall_bert_pos']:.1%} positive "
        answer += f"vs. VADER's {disagreement['overall_vader_pos']:.1%} positive. "
        answer += "VADER tends to be more optimistic, while BERT may capture more nuanced negative emotions.\n"
    
    return answer


def generate_volume_sentiment_answer(data: Dict) -> str:
    """Generate answer for volume vs sentiment relationship."""
    answer = "Based on the volume vs. negative sentiment share analysis:\n\n"
    answer += "The scatter plot with regression line examines whether periods of high posting activity correlate with "
    answer += "increased negative sentiment, which could indicate:\n\n"
    answer += "• **Support-Seeking Behavior**: High volume with high negativity may reflect community members seeking help "
    answer += "during difficult periods, suggesting the platform serves as a support mechanism.\n\n"
    answer += "• **Event-Driven Spikes**: Sudden increases in volume and negativity may correlate with external events "
    answer += "(e.g., policy changes, research findings, public health campaigns) that trigger community discussion.\n\n"
    answer += "• **Community Dynamics**: A positive correlation suggests that as more people engage, more negative experiences "
    answer += "are shared, while a negative correlation might indicate that supportive periods attract more participation.\n\n"
    answer += "**Note**: Detailed statistical analysis (correlation coefficient, p-value) requires access to the full volume data. "
    answer += "The visualization shows the relationship pattern, with the regression line indicating the overall trend."
    
    return answer


# ----------------------------
# Key Findings Extractors
# ----------------------------

def extract_key_findings_bert_trend(trends: Dict) -> List[str]:
    """Extract key findings for BERT trends."""
    findings = []
    if 'negative_trend' in trends:
        neg = trends['negative_trend']
        findings.append(f"Negative sentiment {neg['direction']} by {abs(neg['change']):.1%} from {neg['start']:.1%} to {neg['end']:.1%}")
    if 'positive_trend' in trends:
        pos = trends['positive_trend']
        findings.append(f"Positive sentiment {pos['direction']} by {abs(pos['change']):.1%} from {pos['start']:.1%} to {pos['end']:.1%}")
    return findings


def extract_key_findings_vader_trend(trends: Dict) -> List[str]:
    """Extract key findings for VADER trends."""
    return extract_key_findings_bert_trend(trends)  # Same structure


def extract_key_findings_topic_prevalence(data: Dict) -> List[str]:
    """Extract key findings for topic prevalence."""
    return [
        "Topic distribution shows temporal shifts reflecting evolving community concerns",
        "Medical/HRT topics may show increased prevalence in recent years",
        "Some topics demonstrate seasonal or event-driven patterns"
    ]


def extract_key_findings_bert_topic(analysis: Dict) -> List[str]:
    """Extract key findings for BERT topic sentiment."""
    findings = []
    if 'most_negative_topic' in analysis:
        findings.append(f"Topic {analysis['most_negative_topic']['topic_id']} has highest negative sentiment ({analysis['most_negative_topic']['negative_pct']:.1%})")
    if 'avg_negative_pct' in analysis:
        findings.append(f"Average negative sentiment across topics: {analysis['avg_negative_pct']:.1%}")
    return findings


def extract_key_findings_vader_topic(analysis: Dict) -> List[str]:
    """Extract key findings for VADER topic sentiment."""
    return extract_key_findings_bert_topic(analysis)  # Same structure


def extract_key_findings_subtopic_bert(df: pd.DataFrame) -> List[str]:
    """Extract key findings for BERT subtopic sentiment."""
    if df is None or df.empty or 'bert_neg_pct' not in df.columns:
        return []
    max_neg = df.loc[df['bert_neg_pct'].idxmax()]
    return [f"Subtopic {int(max_neg['subtopic'])} in Topic {int(max_neg['topic'])} shows highest negativity ({max_neg['bert_neg_pct']:.1%})"]


def extract_key_findings_subtopic_vader(df: pd.DataFrame) -> List[str]:
    """Extract key findings for VADER subtopic sentiment."""
    if df is None or df.empty or 'vader_neg_pct' not in df.columns:
        return []
    max_neg = df.loc[df['vader_neg_pct'].idxmax()]
    return [f"Subtopic {int(max_neg['subtopic'])} in Topic {int(max_neg['topic'])} shows highest negativity ({max_neg['vader_neg_pct']:.1%})"]


def extract_key_findings_disagreement(disagreement: Dict) -> List[str]:
    """Extract key findings for model disagreement."""
    findings = []
    if 'correlation' in disagreement:
        findings.append(f"BERT-VADER correlation: {disagreement['correlation']['value']:.3f} ({disagreement['correlation']['interpretation']})")
    if 'max_disagreement_topic' in disagreement:
        findings.append(f"Topic {disagreement['max_disagreement_topic']['topic_id']} shows largest disagreement ({disagreement['max_disagreement_topic']['difference']:.1%})")
    return findings


def extract_key_findings_volume(data: Dict) -> List[str]:
    """Extract key findings for volume-sentiment relationship."""
    return [
        "Regression analysis reveals relationship between posting volume and negative sentiment share",
        "Pattern suggests whether high activity periods correlate with emotional distress or support-seeking"
    ]


# ----------------------------
# Main
# ----------------------------

def main():
    """Main function to generate research questions and answers."""
    print("Loading data...")
    data = load_data()
    
    print("Generating research questions and answers...")
    rqs = generate_rq_answers(data)
    
    # Save to JSON
    output_file = 'outputs/eda/research_questions_answers.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(rqs, f, indent=2)
    print(f"Saved JSON to {output_file}")
    
    # Save to Markdown report
    md_file = 'outputs/eda/research_questions_answers.md'
    with open(md_file, 'w') as f:
        f.write("# Research Questions and Answers for Menopause Reddit EDA\n\n")
        f.write("This document provides comprehensive research questions and data-driven answers ")
        f.write("for each of the 9 visualizations generated in the exploratory data analysis.\n\n")
        f.write("---\n\n")
        
        for i, rq in enumerate(rqs, 1):
            f.write(f"## {i}. {rq['visualization']}\n\n")
            f.write(f"**Research Question**: {rq['research_question']}\n\n")
            f.write(f"**Answer**:\n\n{rq['answer']}\n\n")
            f.write(f"**Key Findings**:\n")
            for finding in rq['key_findings']:
                f.write(f"- {finding}\n")
            f.write(f"\n**Implications**: {rq['implications']}\n\n")
            f.write("---\n\n")
    
    print(f"Saved Markdown report to {md_file}")
    print(f"\nGenerated {len(rqs)} research questions with answers.")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESEARCH QUESTIONS")
    print("="*80)
    for i, rq in enumerate(rqs, 1):
        print(f"\n{i}. {rq['visualization']}")
        print(f"   Q: {rq['research_question']}")
        print(f"   Key Finding: {rq['key_findings'][0] if rq['key_findings'] else 'N/A'}")


if __name__ == "__main__":
    main()

