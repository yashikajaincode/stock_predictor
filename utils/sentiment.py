from transformers import pipeline
from typing import List, Union

# Load sentiment analysis pipeline (uses distilbert-base-uncased-finetuned-sst-2-english by default)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(texts: Union[str, List[str]]) -> List[dict]:
    """
    Analyze sentiment of a string or list of strings.
    Returns a list of dicts with 'label' and 'score'.
    """
    if isinstance(texts, str):
        texts = [texts]
    results = sentiment_pipeline(texts)
    return results 
 