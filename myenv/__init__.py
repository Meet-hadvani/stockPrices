import subprocess

if __name__ == "__main__":
    print("Running stockhistory.py to download stock data and save CSVs...")
    subprocess.run(['python', 'stockhistory.py'], check=True)
    
    print("Running stockpredict.py to load CSVs and predict stock prices...")
    subprocess.run(['python', 'stockpredict.py'], check=True)
