# main.py

# ────────────────────────────────────────
# ✨ SUPPRESS CLUTTERED OUTPUT
# ────────────────────────────────────────
import os
import logging
import warnings

# 1. Suppress TensorFlow logs & messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Suppress UserWarnings, FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# 3. Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel(logging.FATAL)

# 4. Suppress PyTorch logging
import torch
logging.getLogger("torch").setLevel(logging.ERROR)

# ────────────────────────────────────────
# 📦 IMPORTS
# ────────────────────────────────────────
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Flatten, Reshape
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ────────────────────────────────────────
# 🔹 PATH SETUP
# ────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'data', 'large_synthetic_fraud_data.csv')

# ────────────────────────────────────────
# 🔹 STEP 1: Load Dataset
# ────────────────────────────────────────
print("Step 1: Loading Dataset")
data = pd.read_csv(csv_path)
print("Dataset loaded successfully!")

# ────────────────────────────────────────
# 🔹 STEP 2: Preprocessing
# ────────────────────────────────────────
print("Step 2: Preprocessing")

def preprocess(data):
    merchant_encoder = OneHotEncoder(sparse_output=False)
    merchant_encoded = merchant_encoder.fit_transform(data[['MerchantCat']])
    trans_encoder = OneHotEncoder(sparse_output=False)
    trans_encoded = trans_encoder.fit_transform(data[['TransType']])

    for i, cat in enumerate(merchant_encoder.categories_[0]):
        data[f'MerchantCat_{cat}'] = merchant_encoded[:, i]
    for i, typ in enumerate(trans_encoder.categories_[0]):
        data[f'TransType_{typ}'] = trans_encoded[:, i]

    data = data.drop(columns=['MerchantCat', 'TransType', 'Timestamp', 'UserID', 'LinkedUserID', 'LinkType'])

    scaler = MinMaxScaler()
    data[['TransactionAmt', 'KeyHoldTime', 'SwipeSpeed', 'PressureVar']] = scaler.fit_transform(
        data[['TransactionAmt', 'KeyHoldTime', 'SwipeSpeed', 'PressureVar']]
    )
    return data, merchant_encoder, trans_encoder, scaler

data, merchant_encoder, trans_encoder, scaler = preprocess(data)
print("Preprocessing completed!")

# ────────────────────────────────────────
# 🔹 STEP 3: Train-Test Split
# ────────────────────────────────────────
print("Step 3: Splitting the data")
X = data.drop('isFraud', axis=1).values
y = data['isFraud'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

class_weights_arr = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights_arr))
print("Class weights:", class_weight_dict)

# ────────────────────────────────────────
# 🔹 STEP 4: Build CNN Model
# ────────────────────────────────────────
print("Step 4: Building CNN Model")
def build_cnn(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn = build_cnn((X_train.shape[1], 1))
cnn.fit(
    X_train.reshape(-1, X_train.shape[1], 1), y_train,
    epochs=25,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=2
)

# ────────────────────────────────────────
# 🔹 STEP 5: Build LSTM Model
# ────────────────────────────────────────
print("Step 5: Building LSTM Model")
def build_lstm(input_shape):
    model = Sequential([
        Reshape((input_shape[0], 1), input_shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm = build_lstm((X_train.shape[1],))
lstm.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=2
)

# ────────────────────────────────────────
# 🔹 STEP 6: Prepare Graph Data for GNN
# ────────────────────────────────────────
print("Step 6: Preparing Graph Data for GNN")
full_data = pd.read_csv(csv_path)
G = nx.barabasi_albert_graph(len(full_data), 3)

def get_node_features(df, merchant_encoder, trans_encoder, scaler):
    features_list = []
    for idx, row in df.iterrows():
        merchant_cat_enc = merchant_encoder.transform([[row['MerchantCat']]]).flatten()
        trans_type_enc = trans_encoder.transform([[row['TransType']]]).flatten()
        numeric_feats = scaler.transform([[row['TransactionAmt'], row['KeyHoldTime'], row['SwipeSpeed'], row['PressureVar']]]).flatten()
        features = np.concatenate([numeric_feats, merchant_cat_enc, trans_type_enc])
        features_list.append(features)
    return np.array(features_list, dtype=np.float32)

node_features = get_node_features(full_data, merchant_encoder, trans_encoder, scaler)
print(f"Node features shape: {node_features.shape}")

edge_index = torch.tensor(list(G.edges)).t().contiguous()
data_gnn = Data(
    x=torch.tensor(node_features),
    edge_index=edge_index,
    y=torch.tensor(full_data['isFraud'].values, dtype=torch.float).view(-1, 1)
)

train_mask = torch.zeros(len(full_data), dtype=torch.bool)
test_mask = torch.zeros(len(full_data), dtype=torch.bool)
train_indices = list(range(len(X_train)))
test_indices = list(range(len(X_train), len(full_data)))
train_mask[train_indices] = True
test_mask[test_indices] = True

data_gnn.train_mask = train_mask
data_gnn.test_mask = test_mask
print("Graph data prepared for GNN!")

# ────────────────────────────────────────
# 🔹 STEP 7: Define GNN Model
# ────────────────────────────────────────
print("Step 7: Building GNN Model")
class GNNModel(nn.Module):
    def __init__(self, in_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=4)
        self.fc = nn.Linear(32 * 4, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)

gnn = GNNModel(in_channels=node_features.shape[1])

# ────────────────────────────────────────
# 🔹 STEP 8: Train GNN Model
# ────────────────────────────────────────
print("Step 8: Training GNN Model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn.to(device)
data_gnn = data_gnn.to(device)

optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005)
criterion = nn.BCELoss()

gnn.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = gnn(data_gnn.x, data_gnn.edge_index)
    loss = criterion(out[data_gnn.train_mask], data_gnn.y[data_gnn.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ────────────────────────────────────────
# 🔹 STEP 9: Evaluate CNN Model
# ────────────────────────────────────────
print("Step 9: Evaluating CNN Model")
cnn_pred = (cnn.predict(X_test.reshape(-1, X_test.shape[1], 1)) > 0.5).astype(int)
print("CNN Model Performance:")
print(classification_report(y_test, cnn_pred))

# ────────────────────────────────────────
# 🔹 STEP 10: Evaluate LSTM Model
# ────────────────────────────────────────
print("Step 10: Evaluating LSTM Model")
lstm_pred = (lstm.predict(X_test) > 0.5).astype(int)
print("LSTM Model Performance:")
print(classification_report(y_test, lstm_pred))

# ────────────────────────────────────────
# 🔹 STEP 11: Evaluate GNN Model
# ────────────────────────────────────────
print("Step 11: Evaluating GNN Model")
gnn.eval()
with torch.no_grad():
    out = gnn(data_gnn.x, data_gnn.edge_index)
    preds = (out[data_gnn.test_mask] > 0.5).float().cpu()
    labels = data_gnn.y[data_gnn.test_mask].cpu()
    print(classification_report(labels, preds))

print("Training and evaluation complete!")

# ────────────────────────────────────────
# 🔹 STEP 12: Predict Single Transaction
# ────────────────────────────────────────
print("\nStep 12: Single Transaction Fraud Prediction")
def predict_single_transaction(features):
    cnn_prob = cnn.predict(features.reshape(1, -1, 1))[0][0]
    lstm_prob = lstm.predict(features.reshape(1, -1))[0][0]
    gnn_prob = out[0].item()
    combined_prob = (cnn_prob + lstm_prob + gnn_prob) / 3
    return 1 if combined_prob >= 0.5 else 0, combined_prob

sample_index = 0
sample_features = X_test[sample_index]
fraud_label, fraud_prob = predict_single_transaction(sample_features)

print(f"Fraud Prediction: {'Fraud' if fraud_label else 'Non-Fraud'}")
print(f"Prediction Probability: {fraud_prob:.4f}")
