# stockhistory.py
import yfinance as yf
import os

# List of stock symbols
symbols = ['ONCO', 'CNEY', 'TNXP', 'APLD', 'KTTA']

# Folder to save the CSV files int o data folder
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Download stock data and save it as CSV
def download_stock_data():
    start_date = '2019-01-01'
    end_date = '2024-10-01'
    
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        filename = os.path.join(data_folder, f'{symbol}_stock_data.csv')
        stock_data.to_csv(filename)

if __name__ == "__main__":
    download_stock_data()
