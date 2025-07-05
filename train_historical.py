import pandas as pd
from utils.sentiment import analyze_sentiment
from utils.predictor import prepare_features, train_lstm, predict_lstm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load historical stock price data
stock_df = pd.read_csv('data/all_stocks_5yr.csv', parse_dates=['date'])

# 2. Load only the first 1000 news headlines for even faster processing
news_df = pd.read_csv('data/RedditNews.csv', parse_dates=['Date']).head(1000)

# 3. Sentiment analysis on headlines
print('Analyzing sentiment for news headlines...')
news_df['sentiment'] = news_df['News'].apply(lambda x: analyze_sentiment(x)[0]['score'] if isinstance(x, str) and x else 0)

# 4. Aggregate sentiment by date
daily_sentiment = news_df.groupby('Date')['sentiment'].mean().reset_index()

# 5. Process all stocks in the dataset
unique_stocks = stock_df['Name'].unique()
results = []

for stock in unique_stocks:
    print(f'\nProcessing stock: {stock}')
    stock_data = stock_df[stock_df['Name'] == stock].copy()
    # Standardize column names to title case
    stock_data = stock_data.rename(columns={
        'date': 'Date',
        'close': 'Close',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume'
    })
    merged = pd.merge(stock_data, daily_sentiment, on='Date', how='left').fillna(0)
    merged.set_index('Date', inplace=True)
    # Skip stocks without 'Close' price data
    if 'Close' not in merged.columns or merged['Close'].isnull().all():
        print(f'Skipping {stock}: No Close price data.')
        results.append({'stock': stock, 'prediction': 'N/A', 'score': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None})
        continue
    X, y = prepare_features(merged, merged['sentiment'])
    print(f'Feature shape: {X.shape}, Labels shape: {y.shape}')
    if len(X) > 10:
        # Split into train/test (80/20, by time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
        print('Training LSTM model...')
        model = train_lstm(X_train, y_train, input_size=2, epochs=20)
        # Predict on test set for metrics
        y_pred_probs = predict_lstm(model, X_test)
        y_pred = (y_pred_probs > 0.5).astype(int)
        # Calculate metrics on test set
        if len(y_test) > 0:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            accuracy = precision = recall = f1 = None
        # Predict last window
        last_X = X[-1:]
        pred = predict_lstm(model, last_X)[0]
        print(f'Prediction for next day: {"UP" if pred > 0.5 else "DOWN"} ({pred:.2f})')
        print(f'Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        results.append({
            'stock': stock,
            'prediction': 'UP' if pred > 0.5 else 'DOWN',
            'score': float(pred),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    else:
        print('Not enough data to train the model.')
        results.append({'stock': stock, 'prediction': 'N/A', 'score': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None})

# Optionally, save results to a CSV
results_df = pd.DataFrame(results)

# Compute overall (macro) average metrics, excluding N/A
valid = results_df.dropna(subset=['accuracy', 'precision', 'recall', 'f1'])
overall_accuracy = valid['accuracy'].mean() if not valid.empty else None
overall_precision = valid['precision'].mean() if not valid.empty else None
overall_recall = valid['recall'].mean() if not valid.empty else None
overall_f1 = valid['f1'].mean() if not valid.empty else None

print('\nOverall Performance Across All Stocks:')
print(f'Accuracy: {overall_accuracy}')
print(f'Precision: {overall_precision}')
print(f'Recall: {overall_recall}')
print(f'F1-score: {overall_f1}')

# Append overall metrics as a summary row in the CSV
summary_row = pd.DataFrame([{
    'stock': 'OVERALL',
    'prediction': '',
    'score': '',
    'accuracy': overall_accuracy,
    'precision': overall_precision,
    'recall': overall_recall,
    'f1': overall_f1
}])
results_df = pd.concat([results_df, summary_row], ignore_index=True)
results_df.to_csv('data/lstm_predictions_all_stocks.csv', index=False)
print('\nAll stock predictions and overall metrics saved to data/lstm_predictions_all_stocks.csv')