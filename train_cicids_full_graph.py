import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
from torch_geometric.nn import GCNConv
from itertools import combinations


# Config
BASE_PATH = './archive'  # folder where CICIDS csv files are located
BATCH_SIZE = 128
HIDDEN_DIM = 64
DROPOUT = 0.5
EPOCHS = 50
LR = 5e-5
DO_HPO = True
HPO_POP = 6
HPO_ITERS = 5
HPO_EPOCHS = 10
HPO_SUBSAMPLE = 0.5
HPO_VAL_SPLIT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 5

# Enable cuDNN autotuner when using CUDA for potential speedups
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# Load CSVs
csv_files = []
if os.path.isdir(BASE_PATH):
    csv_files = glob.glob(os.path.join(BASE_PATH, '*.csv'))

if not csv_files:
    print(f"No CSV files found in {BASE_PATH}. Exiting.")
    raise SystemExit(1)

print(f"Found {len(csv_files)} CSV files. Loading a subset to keep memory low.")
# For safety, load first N rows from each file (adjust as needed)
dfs = []
for f in csv_files:
    # Try multiple encodings and skip bad lines to robustly read various CICIDS CSV files
    try:
        df_tmp = pd.read_csv(f, encoding='utf-8', engine='python', on_bad_lines='skip')
        dfs.append(df_tmp)
        print(f"Loaded {f} with utf-8: {df_tmp.shape}")
    except Exception:
        try:
            df_tmp = pd.read_csv(f, encoding='latin1', engine='python', on_bad_lines='skip')
            dfs.append(df_tmp)
            print(f"Loaded {f} with latin1: {df_tmp.shape}")
        except Exception as e:
            print(f"Failed to read {f} with utf-8 and latin1: {e}")

if not dfs:
    raise SystemExit("No dataframes loaded")

df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

# Detect label column
label_col = None
for candidate in ['Label', ' label', 'Attack', 'attack', 'Flow Duration']:
    if candidate in df.columns:
        label_col = candidate
        break
if label_col is None:
    # try common alternatives
    for c in df.columns:
        if c.strip().lower() == 'label':
            label_col = c
            break

if label_col is None:
    raise SystemExit('Label column not found in CICIDS files')

print(f"Using label column: {label_col}")

# Keep only numeric features
numeric_df = df.select_dtypes(include=[np.number]).copy()
if label_col in numeric_df.columns:
    # label might be numeric; if so, remove from features and keep original label from df
    numeric_df = numeric_df.drop(columns=[label_col])

print(f"\nMemory optimization and feature cleaning...")
memory_before = numeric_df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage before optimization: {memory_before:.2f} MB")

# Remove zero-variance features first
variance = numeric_df.var()
zero_var_cols = variance[variance == 0].index.tolist()
if zero_var_cols:
    print(f"Found {len(zero_var_cols)} zero-variance features, removing them")
    numeric_df = numeric_df.drop(columns=zero_var_cols)

# Handle missing values and sanitize numeric features (replace infinities, fill NaN, clip extremes)
numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop columns that are entirely NaN
all_nan_cols = numeric_df.columns[numeric_df.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN numeric columns: {all_nan_cols}")
    numeric_df = numeric_df.drop(columns=all_nan_cols)

# Fill remaining NaNs with column median and clip extreme outliers per-column
for col in numeric_df.columns:
    col_vals = numeric_df[col]
    if col_vals.isna().all():
        # nothing left in this column
        numeric_df.drop(columns=[col], inplace=True)
        continue
    median = col_vals.median()
    if np.isnan(median):
        median = 0.0
    numeric_df[col].fillna(median, inplace=True)
    try:
        p99 = np.nanpercentile(np.abs(col_vals.dropna()), 99.9)
        cap = max(p99, 1e6)
    except Exception:
        cap = 1e6
    # Clip to reasonable range to avoid overflow in float32
    numeric_df[col] = numeric_df[col].clip(-cap, cap)

# Optimize data types to reduce memory usage (following notebook approach)
print("\nOptimizing data types...")
for col in numeric_df.columns:
    col_min = numeric_df[col].min()
    col_max = numeric_df[col].max()
    
    # Integer optimization
    if numeric_df[col].dtype in ['int64', 'int32']:
        if col_min >= 0 and col_max <= 255:
            numeric_df[col] = numeric_df[col].astype('uint8')
        elif col_min >= 0 and col_max <= 65535:
            numeric_df[col] = numeric_df[col].astype('uint16')
        elif col_min >= -32768 and col_max <= 32767:
            numeric_df[col] = numeric_df[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            numeric_df[col] = numeric_df[col].astype('int32')
    
    # Float optimization - always convert float64 to float32
    elif numeric_df[col].dtype == 'float64':
        numeric_df[col] = numeric_df[col].astype('float32')

memory_after = numeric_df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage after optimization: {memory_after:.2f} MB")
print(f"Memory reduced by {(1 - memory_after/memory_before)*100:.1f}%")

# Encode labels - drop rows with NaN labels instead of converting to 'Unknown'
print(f"\nOriginal dataset size: {len(df)}")
labels = df[label_col].astype(str)
# Remove rows where label is 'nan' string or empty
valid_mask = ~labels.isin(['nan', 'None', '', 'NaN'])
print(f"Rows with invalid labels: {(~valid_mask).sum()}")
labels = labels[valid_mask]
numeric_df = numeric_df[valid_mask].reset_index(drop=True)
labels = labels.reset_index(drop=True)
print(f"Dataset size after removing invalid labels: {len(labels)}")

# Print class distribution
print("\nClass distribution:")
class_counts = labels.value_counts().sort_values(ascending=False)
for cls, cnt in class_counts.items():
    print(f"  {cls}: {cnt}")

le = LabelEncoder()
y = le.fit_transform(labels)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(numeric_df.values.astype(np.float32))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size before resampling: {len(y_train)}")
print("Training class distribution before resampling:")
train_counter = Counter(y_train)
for cls_idx in sorted(train_counter.keys()):
    print(f"  {le.classes_[cls_idx]}: {train_counter[cls_idx]}")

# Apply hybrid resampling: SMOTE + Random Under-Sampling (following notebook approach)
# This prevents both: 1) overfitting to synthetic samples, 2) majority class dominance
try:
    # Calculate target samples following notebook's strategy
    majority_class = max(train_counter, key=train_counter.get)
    majority_count = train_counter[majority_class]
    target_minority = int(majority_count * 0.1)  # Over-sample minorities to 10% of majority
    
    print(f"\nHybrid resampling strategy:")
    print(f"  Majority class: {le.classes_[majority_class]} ({majority_count:,} samples)")
    print(f"  Target for minorities: {target_minority:,} samples (10% of majority)")
    
    # SMOTE over-sampling for minorities
    sampling_strategy_over = {
        cls: max(count, target_minority)
        for cls, count in train_counter.items()
        if cls != majority_class and count < target_minority
    }
    
    # Under-sample majority to 5x the target minority count
    target_majority = int(target_minority * 5)
    sampling_strategy_under = {majority_class: target_majority}
    
    print(f"  Target for majority after under-sampling: {target_majority:,} samples")
    
    # Create hybrid pipeline
    k_neighbors = min(5, min(train_counter.values()) - 1) if min(train_counter.values()) > 1 else 1
    over = SMOTE(sampling_strategy=sampling_strategy_over, random_state=42, k_neighbors=k_neighbors)
    under = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
    pipeline = ImbPipeline(steps=[('over', over), ('under', under)])
    
    # Apply resampling
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    
    print(f"\nTraining set size after hybrid resampling: {len(y_train)}")
    print("Training class distribution after resampling:")
    train_counter_after = Counter(y_train)
    for cls_idx in sorted(train_counter_after.keys()):
        change = train_counter_after[cls_idx] - train_counter.get(cls_idx, 0)
        print(f"  {le.classes_[cls_idx]}: {train_counter_after[cls_idx]} ({change:+d})")
except Exception as e:
    print(f"Hybrid resampling failed: {e}. Continuing without resampling.")

class AttackDataset(Dataset):
    def __init__(self, X, y,window_size=WINDOW_SIZE):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.window_size = window_size
        # Calculate number of complete windows
        self.num_samples = len(self.y) - self.window_size + 1
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        window_start = idx
        window_end = window_start + self.window_size
        x_window = self.X[window_start:window_end]
        y_window = self.y[window_start:window_end]
        # Return the last label in the window as the target
        return x_window, y_window[-1]       
# Create WeightedRandomSampler for balanced batch sampling
train_dataset = AttackDataset(X_train, y_train, window_size=WINDOW_SIZE)
# Calculate sample weights: inverse of class frequency
class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weight_per_class = 1. / class_sample_count
sample_weights = np.array([weight_per_class[t] for t in y_train])
sample_weights = torch.from_numpy(sample_weights).float()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True, num_workers=4)
test_loader = DataLoader(AttackDataset(X_test, y_test,window_size=WINDOW_SIZE), batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

# Model (stacked bidirectional GRU + GCN)
class BiGRU_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, window_size=5, dropout=0.5):
        super(BiGRU_GCN, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.gnn1 = GCNConv(hidden_dim * 2, 128)
        self.gnn2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, output_dim)
        self.window_size = window_size

    def create_fully_connected_grpah(self,num_nodes,device):
        if(num_nodes<2):
            edge_index = torch.tensor([[0],[0]],dtype=torch.long,device=device)
        else:
            edge_list = list(combinations(range(num_nodes),2))
            edge_index = torch.tensor([
                [src for src, dst in edge_list] + [dst for src, dst in edge_list],
                [dst for src, dst in edge_list] + [src for src, dst in edge_list],
            ],dtype =torch.long, device=device)
        return edge_index

    def forward(self, x):
        # x: (batch, features) -> treat features as single-step sequence
        batch_size, window_size,num_feature = x.shape
        x = x.view(batch_size*window_size, num_feature)
        out1, _ = self.gru(x.unsqueeze(1))
        out2, _ = self.gru2(out1)
        seq_out = out2.view(batch_size,window_size,-1)
        # simple static graph (small placeholder) - build a fully-connected chain of length = seq_out.size(0)
        seq_out = seq_out [:,-1,:].squeeze(1)  # shape: (batch_size, hidden_dim*2)
        edge_index = self.create_fully_connected_grpah(seq_out.size(0),seq_out.device)

        g1 = self.gnn1(seq_out, edge_index)
        g1 = self.dropout(g1)
        g2 = self.gnn2(g1, edge_index)
        g2 = self.dropout(g2)
        out = self.fc(g2)
        return out

def _train_eval_hpo(X_data, y_data, input_dim, output_dim, lr, hidden_dim, dropout):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_data, y_data, test_size=HPO_VAL_SPLIT, random_state=42, stratify=y_data
    )

    train_ds = AttackDataset(X_tr, y_tr)
    val_ds = AttackDataset(X_val, y_val)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)

    model = BiGRU_GCN(input_dim, hidden_dim, output_dim, window_size=WINDOW_SIZE).to(DEVICE)
    class_ids = np.arange(output_dim)
    class_weights = compute_class_weight('balanced', classes=class_ids, y=y_tr)
    class_weights = np.nan_to_num(class_weights, nan=0.0, posinf=0.0, neginf=0.0)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(HPO_EPOCHS):
        model.train()
        for Xb, yb in train_ld:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for Xb, yb in val_ld:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            out = model(Xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return (100.0 * correct / total) if total > 0 else 0.0

def _sbo_optimize(X_data, y_data, input_dim, output_dim):
    bounds = np.array([
        [-5.0, -3.3],  # log10(lr) -> 1e-5 to ~5e-4
        [32.0, 256.0], # hidden_dim
        [0.1, 0.6],    # dropout
    ], dtype=np.float32)

    def decode(x):
        lr = 10 ** float(x[0])
        hidden = int(round(float(x[1]) / 16.0) * 16)
        hidden = int(np.clip(hidden, 32, 256))
        drop = float(np.clip(x[2], 0.1, 0.6))
        return lr, hidden, drop

    def clip(x):
        return np.clip(x, bounds[:, 0], bounds[:, 1])

    print(f"  [Init] Evaluating initial population of {HPO_POP} candidates...")
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(HPO_POP, bounds.shape[0]))
    fitness = []
    for i in range(HPO_POP):
        lr, hidden, drop = decode(pop[i])
        acc = _train_eval_hpo(X_data, y_data, input_dim, output_dim, lr, hidden, drop)
        fitness.append(acc)
        print(f"    Eval {i+1}/{HPO_POP}: acc={acc:.2f}% | lr={lr:.6f}, hidden={hidden}, drop={drop:.2f}")
    fitness = np.array(fitness, dtype=np.float32)

    best_idx = int(np.argmax(fitness))
    best_pos = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    print(f"  [Init] Best initial fitness: {best_fit:.2f}%\n")

    for iter_num in range(HPO_ITERS):
        print(f"  [Iteration {iter_num+1}/{HPO_ITERS}]")
        for i in range(HPO_POP):
            r1, r2 = np.random.rand(2)
            rand_pos = np.random.uniform(bounds[:, 0], bounds[:, 1])
            new_pos = pop[i] + r1 * (best_pos - np.abs(pop[i])) + r2 * (rand_pos - pop[i])
            new_pos = clip(new_pos)

            lr, hidden, drop = decode(new_pos)
            acc = _train_eval_hpo(X_data, y_data, input_dim, output_dim, lr, hidden, drop)
            if acc > fitness[i]:
                pop[i] = new_pos
                fitness[i] = acc
                status = "✓ IMPROVED" if acc > best_fit else "↑ updated"
                print(f"    Agent {i+1}/{HPO_POP}: {status} acc={acc:.2f}% | lr={lr:.6f}, hidden={hidden}, drop={drop:.2f}")
                if acc > best_fit:
                    best_fit = float(acc)
                    best_pos = new_pos.copy()
                    print(f"    >>> NEW BEST: {best_fit:.2f}%")
        print()

    best_lr, best_hidden, best_drop = decode(best_pos)
    print(f"  [Complete] SBO optimization finished!")
    print(f"  Final best fitness: {best_fit:.2f}%")
    return best_lr, best_hidden, best_drop, best_fit

input_dim = X_train.shape[1]
output_dim = len(le.classes_)
if DO_HPO:
    np.random.seed(42)
    hpo_n = min(len(y_train), max(2000, int(len(y_train) * HPO_SUBSAMPLE)))
    hpo_idx = np.random.choice(len(y_train), hpo_n, replace=False)
    X_hpo = X_train[hpo_idx]
    y_hpo = y_train[hpo_idx]
    print(f"\nRunning Secretary Bird Optimization (SBO) for HPO on {hpo_n} samples...")
    best_lr, best_hidden, best_drop, best_fit = _sbo_optimize(X_hpo, y_hpo, input_dim, output_dim)
    LR = best_lr
    HIDDEN_DIM = best_hidden
    DROPOUT = best_drop
    print(
        f"SBO best params -> lr: {LR:.6f}, hidden_dim: {HIDDEN_DIM}, dropout: {DROPOUT:.2f} "
        f"(val acc: {best_fit:.2f}%)"
    )

model = BiGRU_GCN(input_dim, HIDDEN_DIM, output_dim, dropout=DROPOUT).to(DEVICE)
print(f"Using device: {DEVICE}")
try:
    param_dev = next(model.parameters()).device
    print(f"Model parameters device: {param_dev}")
except StopIteration:
    print("Model has no parameters to check device.")

# Define focal loss to handle class imbalance better than standard cross-entropy
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Compute class weights from training labels
class_ids = np.arange(output_dim)
class_weights = compute_class_weight('balanced', classes=class_ids, y=y_train)
class_weights = np.nan_to_num(class_weights, nan=0.0, posinf=0.0, neginf=0.0)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Use focal loss with gamma=2.0 to focus more on hard examples
criterion = FocalLoss(alpha=weight_tensor, gamma=2.0)
print(f"\nUsing Focal Loss with gamma=2.0 and class weights")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Early stopping / checkpointing
best_acc = 0.0
patience = 10  # Increased patience for 50 epoch training
patience_counter = 0
best_epoch = 0
best_model_path = 'cicids_bigru_gcn_model_best.pth'

# Training loop
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    correct = 0
    for Xb, yb in train_loader:
        # transfer to device with non-blocking to overlap CPU->GPU copies
        Xb = Xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        # print device information for the first batch after transfer
        if total_loss == 0 and correct == 0:
            try:
                print(f"First batch tensor device after transfer: {Xb.device}")
            except Exception:
                pass
        optimizer.zero_grad()
        out = model(Xb)
        yb_last = yb[:,-1]

        loss = criterion(out, yb_last)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb_last).sum().item()
    acc = 100 * correct / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.2f}%")
    # Evaluate on test set and compute per-class accuracy for this epoch
    model.eval()
    y_true_epoch = []
    y_pred_epoch = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            preds = out.argmax(dim=1).cpu().numpy()
            yb_last = yb[:,-1].numpy()
            y_pred_epoch.extend(preds.tolist())
            y_true_epoch.extend(yb_last.tolist())

    y_true_epoch = np.array(y_true_epoch)
    y_pred_epoch = np.array(y_pred_epoch)
    print(f"Per-class accuracy after epoch {epoch}:")
    for idx, cls in enumerate(le.classes_):
        mask = (y_true_epoch == idx)
        total = int(mask.sum())
        if total == 0:
            print(f"  {cls}: no samples in test set")
            continue
        correct = int((y_pred_epoch[mask] == idx).sum())
        acc_cls = 100.0 * correct / total
        print(f"  {cls}: {acc_cls:.2f}% ({correct}/{total})")

    # overall test accuracy and early stopping/checkpointing
    test_total = int(y_true_epoch.size)
    test_correct = int((y_pred_epoch == y_true_epoch).sum()) if test_total > 0 else 0
    test_acc = 100.0 * test_correct / test_total if test_total > 0 else 0.0
    print(f"Test overall accuracy after epoch {epoch}: {test_acc:.2f}% ({test_correct}/{test_total})")
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f" New best model saved: {best_model_path} (epoch {epoch}, acc {best_acc:.2f}%)")
    else:
        patience_counter += 1
        print(f" No improvement. patience {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

# Save artifacts
torch.save(model.state_dict(), 'cicids_bigru_gcn_model.pth')
joblib.dump(le, 'cicids_label_encoder.pkl')
joblib.dump(scaler, 'cicids_scaler.pkl')
print('Saved model and preprocessing artifacts.')
