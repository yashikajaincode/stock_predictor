import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple

class StockSentimentLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(StockSentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)
        return self.sigmoid(out)

def prepare_features(price_df: pd.DataFrame, sentiment_scores: pd.Series, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for LSTM.
    price_df: DataFrame with 'Close' column
    sentiment_scores: Series indexed by date
    window: lookback window for LSTM
    Returns: X, y
    """
    # Merge price and sentiment
    df = price_df[['Close']].copy()
    df['sentiment'] = sentiment_scores.reindex(df.index, fill_value=0)
    df = df.dropna()
    X, y = [], []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:i+window][['Close', 'sentiment']].values
        target = 1 if df.iloc[i+window]['Close'] > df.iloc[i+window-1]['Close'] else 0  # Up or down
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)

def train_lstm(X: np.ndarray, y: np.ndarray, input_size: int, epochs: int = 20, lr: float = 0.001) -> StockSentimentLSTM:
    model = StockSentimentLSTM(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    return model

def predict_lstm(model: StockSentimentLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()
    return preds 