import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class LSTM_GCN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTM_GCN_Model, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.gnn1 = GCNConv(hidden_dim * 2, 128)
        self.gnn2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, output_dim)  

    def forward(self, x):
        gru_out, _ = self.gru(x.unsqueeze(1))
        gru_out2, _ = self.gru2(gru_out)
        lstm_out = gru_out2[:, -1, :]
        edge_index = torch.tensor([[0, 1, 2, 3, 4], 
                           [1, 2, 3, 4, 5]], dtype=torch.long)

        gnn_out1 = self.gnn1(lstm_out, edge_index)
        gnn_out1 = self.dropout(gnn_out1)  
        gnn_out2 = self.gnn2(gnn_out1, edge_index)
        gnn_out2 = self.dropout(gnn_out2)  

        output = self.fc(gnn_out2)  
        return output

label_encoder = joblib.load("label_encoder.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

input_dim = tfidf_vectorizer.max_features + 2
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = LSTM_GCN_Model(input_dim, hidden_dim, output_dim)
import os
if os.path.exists("lstm_gcn_model.pth"):
    try:
        model.load_state_dict(torch.load("lstm_gcn_model.pth"))
        model.eval()
    except Exception as e:
        print(f"Warning: failed to load lstm_gcn_model.pth: {e}\nContinuing with randomly initialized model.")
else:
    print("lstm_gcn_model.pth not found — continuing with randomly initialized model.")

# Network traffic data to test
new_csv_file = "wireshark-test.csv"
data = pd.read_csv(new_csv_file, encoding="latin1")

numeric_cols = ['Time', 'Length']
data_numeric = data[numeric_cols].fillna(0).values.astype(np.float32)
data_tfidf = tfidf_vectorizer.transform(data['Info'].fillna("")).toarray()
X_new = np.hstack((data_numeric, data_tfidf)).astype(np.float32)
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_new_tensor)
    _, predicted = torch.max(outputs, 1)
data['attack_type'] = label_encoder.inverse_transform(predicted.numpy())

# Saving the results to a new file
output_csv_file = "predicted_attack_types.csv"
data.to_csv(output_csv_file, index=False)
print(f"Predictions saved: {output_csv_file}")