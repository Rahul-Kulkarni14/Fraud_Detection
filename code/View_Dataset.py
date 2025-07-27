import pandas as pd
import os

# Get absolute path to the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the dataset
csv_path = os.path.join(base_dir, 'data', 'large_synthetic_fraud_data.csv')

# Load the dataset
data = pd.read_csv(csv_path)

# View first 10 rows
print(data.head(10))

# View summary info
print(data.info())
