import yfinance as yf
import pandas as pd
from newspaper import Article
import requests
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Stock Price Fetching (yfinance) ---
def fetch_stock_data(ticker: str, period: str = '1mo', interval: str = '1d') -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# --- News Scraping (NewsAPI) ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Get NewsAPI key from environment variables

def fetch_newsapi_headlines(query: str, api_key: Optional[str] = NEWSAPI_KEY, n: int = 5) -> List[str]:
    """
    Fetch recent news headlines using NewsAPI.
    """
    if not api_key:
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize={n}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [a['title'] for a in articles]
    return []

# --- News Scraping (newspaper3k) ---
def fetch_news_from_url(url: str) -> str:
    """
    Scrape the main text from a news article URL.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text 