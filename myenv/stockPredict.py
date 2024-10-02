import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

# List of stocks
symbols = ['ONCO', 'CNEY', 'TNXP', 'APLD', 'KTTA']

# Dictionary to store the loaded data for each stock
stock_data_dict = {}
#prediction range is for next 120 days can be changed.
prediction_horizon = 120
model_dict = {}
data_folder = "data"

# Load data from CSV files into a dictionary
for symbol in symbols:
    filename = os.path.join(data_folder, f'{symbol}_stock_data.csv')
    stock_data = pd.read_csv(filename, parse_dates=True, index_col='Date')
    stock_data_dict[symbol] = stock_data


# Train a simple model for each stock and predict the next 4 months
for symbol, stock_data in stock_data_dict.items():
    stock_data['Target'] = stock_data['Close'].shift(-1)
    stock_data = stock_data.dropna()

    X = stock_data[['Close']]
    y = stock_data['Target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    last_close_price = stock_data['Close'].iloc[-1]
    future_dates = pd.date_range(stock_data.index[-1], periods=prediction_horizon + 1, freq='B')[1:]
    
    future_predictions = []
    current_price = np.array([last_close_price]).reshape(-1, 1)
    
    for _ in range(prediction_horizon):
        next_prediction = model.predict(current_price)[0]
        future_predictions.append(next_prediction)
        current_price = np.array([next_prediction]).reshape(-1, 1)
    
    # Save the predictions to the dictionary
    future_df = pd.DataFrame({'Predicted Close': future_predictions}, index=future_dates)
    model_dict[symbol] = {'model': model, 'future_df': future_df}
    
    # Evaluate the model on test data
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Plot the future predictions on matplotlib.
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Historical Prices')
    plt.plot(future_df.index, future_df['Predicted Close'], label='Predicted Prices', linestyle='--')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
