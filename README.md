
# 🧠 DeepFraudNet – Real-Time Banking Fraud Detection System

An intelligent, modular fraud detection system using CNN, LSTM, and Graph Neural Networks (GNNs) to identify suspicious banking transactions in real-time. The system is built to model complex behavioral patterns and transaction sequences, leveraging both temporal and relational features.

---

## 🔍 Overview

**DeepFraudNet** is a hybrid machine learning framework that detects anomalous and fraudulent banking behavior using:
- **Convolutional Neural Networks (CNN)** for behavioral profiling
- **Long Short-Term Memory (LSTM)** for temporal transaction sequence analysis
- **Graph Neural Networks (GNN)** via PyTorch Geometric for relational risk modeling (e.g., between users or accounts)

This model simulates a live banking environment using synthetic datasets and can be extended to real-world deployments for fraud analytics.

---

## 🚀 Features

- 💳 Transaction classification: genuine vs suspicious
- 📊 Real-time data generation and labeling
- 🧬 CNN-based behavioral embeddings
- 🕒 LSTM for temporal dependencies across transactions
- 🕸️ GNN for modeling complex relationships among users/accounts
- 📈 Expandable dataset generator and viewer
- 🧠 Modular codebase for separate model training

---

## ⚙️ Technologies Used

| Model Type | Framework            | Purpose                                |
|------------|----------------------|----------------------------------------|
| CNN        | TensorFlow / Keras   | Behavior-based fraud detection         |
| LSTM       | TensorFlow / Keras   | Sequence learning on transaction time  |
| GNN        | PyTorch Geometric    | Relationship risk and fraud networks   |

---

## 📦 Installation

Install all required packages:

```bash
pip install -r requirements.txt
````

> For GNN support, ensure you have PyTorch + PyTorch Geometric installed correctly.

---

## 🧪 Running the System

```bash
# 1. Generate or update the dataset
python code/generate_dataset.py

# 2. View dataset stats or samples
python code/view_dataset.py

# 3. Run the full detection pipeline
python code/main.py

---

## 🧠 Model Training Insights

* **CNN** is used on tabular data converted to 2D behavioral profiles
* **LSTM** processes transaction sequences per user
* **GNN** models account-to-account transaction graphs

Each model is trained separately and integrated into the final ensemble classifier in `main.py`.

---

## 📈 Evaluation Metrics

* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix per model
* AUC-ROC Curves
* False positive rate reduction through hybrid voting

---

## 🛡️ Real-World Impact

Fraud detection models like DeepFraudNet are crucial for:

* Preventing digital payment scams
* Identifying compromised accounts
* Enhancing customer trust in banking platforms

This research has been peer-reviewed and demonstrates practical fraud detection approaches using modern deep learning techniques.


