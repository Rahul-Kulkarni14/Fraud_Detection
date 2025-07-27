import pandas as pd
import numpy as np
import networkx as nx
import os

# Set random seed for reproducibility
np.random.seed(42)
num_records = 10000

# Transactional Data
transaction_amt = np.random.exponential(scale=100, size=num_records)
timestamps = pd.date_range(start='2025-01-01', periods=num_records, freq='h')
merchant_categories = np.random.choice(['Electronics', 'Groceries', 'Fashion', 'Travel', 'Food'], num_records)
transaction_types = np.random.choice(['Online', 'POS', 'ATM', 'Transfer'], num_records)

# Behavioral Biometrics Data
key_hold_time = np.random.normal(loc=0.15, scale=0.05, size=num_records)
swipe_speed = np.random.normal(loc=1.5, scale=0.3, size=num_records)
pressure_var = np.random.normal(loc=0.5, scale=0.1, size=num_records)

# Graph Data
G = nx.barabasi_albert_graph(num_records, 3)
user_ids = list(G.nodes)
linked_user_ids = [np.random.choice(list(G.neighbors(i)), 1)[0] if len(list(G.neighbors(i))) > 0 else i for i in user_ids]
link_types = np.random.choice(['Friend', 'Colleague', 'Business', 'Family'], num_records)

# Fraud Label
fraudulent = np.random.binomial(1, 0.1, num_records)

# Combine data into DataFrame
dataset = pd.DataFrame({
    'TransactionAmt': transaction_amt,
    'Timestamp': timestamps,
    'MerchantCat': merchant_categories,
    'TransType': transaction_types,
    'KeyHoldTime': key_hold_time,
    'SwipeSpeed': swipe_speed,
    'PressureVar': pressure_var,
    'UserID': user_ids,
    'LinkedUserID': linked_user_ids,
    'LinkType': link_types,
    'isFraud': fraudulent
})

# Dynamically get the directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, 'large_synthetic_fraud_data.csv')

# Save dataset
dataset.to_csv(csv_path, index=False)
print(f"Dataset generated and saved to: {csv_path}")
