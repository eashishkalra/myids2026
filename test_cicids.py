import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class BiGRU_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(BiGRU_GCN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.gnn1 = GCNConv(hidden_dim * 2, 128)
        self.gnn2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        out1, _ = self.gru(x.unsqueeze(1))
        out2, _ = self.gru2(out1)
        seq_out = out2[:, -1, :]
        n = seq_out.size(0)
        if n < 2:
            edge_index = torch.tensor([[0],[0]], dtype=torch.long, device=seq_out.device)
        else:
            src = torch.arange(0, n-1, dtype=torch.long, device=seq_out.device)
            dst = torch.arange(1, n, dtype=torch.long, device=seq_out.device)
            edge_index = torch.stack([torch.cat([src,dst]), torch.cat([dst, src])], dim=0)
        g1 = self.gnn1(seq_out, edge_index)
        g1 = self.dropout(g1)
        g2 = self.gnn2(g1, edge_index)
        g2 = self.dropout(g2)
        out = self.fc(g2)
        return out

# Config
MODEL_PATH = 'cicids_bigru_gcn_model.pth'
LE_PATH = 'cicids_label_encoder.pkl'
SCALER_PATH = 'cicids_scaler.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH) or not os.path.exists(SCALER_PATH):
    raise SystemExit('Model or preprocessing artifacts not found. Run train_cicids.py first.')

le = joblib.load(LE_PATH)
scaler = joblib.load(SCALER_PATH)

# Input CSV to predict
input_csv = 'wireshark-test.csv' if os.path.exists('wireshark-test.csv') else None
if input_csv is None:
    raise SystemExit('No input CSV provided (looking for wireshark-test.csv).')

df = pd.read_csv(input_csv)
# keep numeric columns only
X = df.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
Xs = scaler.transform(X)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiGRU_GCN(Xs.shape[1], 64, len(le.classes_)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

import torch
with torch.no_grad():
    Xt = torch.tensor(Xs, dtype=torch.float32).to(device)
    out = model(Xt)
    preds = out.argmax(dim=1).cpu().numpy()

df['predicted_label'] = le.inverse_transform(preds)
out_csv = 'cicids_predicted.csv'
df.to_csv(out_csv, index=False)
print(f'Predictions written to {out_csv}')
