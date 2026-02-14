import pandas as pd
import os
import requests

def download_bangalore_data(output_path):
    url = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/Bengaluru_House_Data.csv"
    print(f"Downloading Bengaluru House Data from {url}...")
    
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset saved to {output_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    raw_data_path = os.path.join('house-price-prediction', 'data', 'raw', 'bangalore_housing.csv')
    download_bangalore_data(raw_data_path)
