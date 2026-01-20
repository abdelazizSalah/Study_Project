# resnet_experiments.py
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from file_helper_t3 import load_k_fold_results
from labels_helper import deduplicate_folds, encode_labels
from handling_re_bytes_integrated import get_keep_indices_from_fold0_ae


# ---------------------------
# ResNet1D model definition
# ---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) BasicBlock (1D) for ResNet-18
# -----------------------------
class BasicBlock1D(nn.Module):
    """
    ResNet-18 BasicBlock: two Conv1d layers with BN + ReLU between,
    plus an identity/projection skip connection.
    """

    # no expansion for the output channels in BasicBlock
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        '''
        in_channels:  number of input channels
        out_channels: number of output channels
        stride: controls downsampling
        
        '''
        # initialize parent class
        super().__init__()

        # first conv layer with kernel size = 3 and padding = 1 to maintain the sequence length
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        # normalizes each channel across the batch and sequence length
        # it improves training stability and convergence
        self.bn1 = nn.BatchNorm1d(out_channels)

        # second conv layer, forces stride = 1 to perserve resolution inside the block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # second batch normalization layer
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip path: identity if shape matches, otherwise 1x1 projection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        '''
            Forward pass of the BasicBlock1D.
            x: input tensor of shape (B, C_in, M)
            returns: output tensor of shape (B, C_out, M_out)
        
        '''

        # stores the input for skip connection
        identity = x

        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Second conv + BN, and we do not use ReLU here yet
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # applies skip connection to enable the residual learning
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# -----------------------------
# 2) ResNet-18 backbone (1D): 4 stages, each stage has 2 blocks
# -----------------------------
class ResNet18_1D(nn.Module):
    """
    ResNet-18 (1D) layout:
      stem: Conv1d(7) + BN + ReLU + MaxPool
      stage1: 64  (2 blocks)
      stage2: 128 (2 blocks, first block stride=2)
      stage3: 256 (2 blocks, first block stride=2)
      stage4: 512 (2 blocks, first block stride=2)
      head: AdaptiveAvgPool1d(1) -> FC1+ReLU+Dropout -> FC2 (2 logits)

    Optionally concatenates 5 stats features at the embedding stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout_p: float = 0.3,
        use_stats: bool = False, # whether to use the 5 stats features
        stats_dim: int = 5, # number of stats features (if used)
    ):
        super().__init__()
        self.use_stats = use_stats
        self.stats_dim = stats_dim

        # Stem (like classic ResNet, adapted to 1D)
        '''
            Stem are the very first part of the network, applied directly to the raw input
            its main purpose is to: 
                - Quickly extract basic low-level patterns
                - Reduce the input resolution early
                - Convert raw input into a feature representation suitable for deep processing
            For input shape: (B, 1, M)
            after Stem: 
                (B, 64, M/4)
            So the number of channels increases, but the length M decreases to the quarter
        '''
        self.stem_conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 4 stages, 2 blocks each (ResNet-18 pattern)
        '''
            Stages are groups of residual blocks that form the main body of the ResNet
            Each stage: 
                - Has a fixed number of channels
                - Contains multiple residual blocks
                - Learns features at a specific abstraction level
            So we perform the following each stage:
                - Use Residual learning
                - Applies Conv -> BN -> ReLU -> Conv -> BN
                - Refines features rather than just detecting them
            Progression: 
                - Stage 1 -> low-level features
                - Stage 2 -> mid-level features
                - Stage 3 -> high-level features
                - Stage 4 -> very abstract features
        '''
        self.in_planes = 64
        self.layer1 = self._make_layer(out_channels=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

        # Pooling -> embedding
        # convert (B, 512, M) to (B, 512, 1) independent of the input length M
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # FC head (2 FC layers)
        # embedding_dim = 512 after avgpool (because stage4 out_channels=512)
        embedding_dim = 512
        fc1_in = embedding_dim + (stats_dim if use_stats else 0)

        self.fc1 = nn.Linear(fc1_in, hidden_dim)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # init like typical ResNet
        self._init_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = []
        # First block may downsample via stride
        layers.append(BasicBlock1D(self.in_planes, out_channels, stride=stride))
        self.in_planes = out_channels * BasicBlock1D.expansion
        # Remaining blocks keep stride=1
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        '''
            These are the standard normalization for ReLU and BatchNorm networks 
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, stats=None):
        """
        x:     (B, C_in, M)  e.g. (B,1,M) for raw bytes/sequence
        stats: (B, 5) if use_stats=True, else None
        """
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = F.relu(x, inplace=True)
        x = self.stem_pool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive pooling -> (B, 512, 1) -> flatten -> (B, 512)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Optionally concatenate stats features
        if self.use_stats:
            if stats is None:
                raise ValueError("use_stats=True but stats=None was passed to forward().")
            if stats.dim() != 2 or stats.size(1) != self.stats_dim:
                raise ValueError(f"stats must be (B,{self.stats_dim}), got {tuple(stats.shape)}.")
            x = torch.cat([x, stats], dim=1)

        # FC head: FC1 + ReLU + Dropout, then FC2 -> logits
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits




# -----------------------------
# Metrics (same as your CNN file)
# -----------------------------
def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


# -----------------------------
# 5 stats features (optional)
# Sheet requires first 3 + 2 proposed :contentReference[oaicite:1]{index=1}
# Proposed extras here:
#   - Shannon entropy of bytes (captures “randomness/structure”)
#   - Fraction of zero bytes (captures padding / sparsity patterns)
# -----------------------------
def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / float(total)
    return float(-(p * np.log2(p)).sum())


def compute_stats_features(
    X_bytes: np.ndarray,
    pad_value: int = 0,
    assume_padded: bool = True,
) -> np.ndarray:
    """
    X_bytes: (N, M) uint8 or int array, values in [0,255]
    Returns: (N, 5) float32 stats:
      1) total_bytes_before_trim/pad (heuristic if only padded array exists)
      2) count of most frequent byte
      3) count of 2nd most frequent byte
      4) Shannon entropy
      5) fraction of pad_value bytes
    Note: If you truly have original (pre-pad/trim) lengths elsewhere, replace total_bytes accordingly.
    """
    X = X_bytes
    if X.dtype != np.uint8:
        X = X.astype(np.uint8)

    N, M = X.shape
    feats = np.zeros((N, 5), dtype=np.float32)

    for i in range(N):
        row = X[i]

        # "total bytes before trimming/padding" requirement :contentReference[oaicite:2]{index=2}
        # If you only have padded arrays, best practical approximation:
        # count non-pad bytes when padding exists, else M.
        if assume_padded:
            total_bytes = int(np.sum(row != pad_value))
            if total_bytes == 0:
                total_bytes = M
        else:
            total_bytes = M

        counts = np.bincount(row, minlength=256)

        # top-2 frequencies
        top2 = np.sort(counts)[-2:]
        top1_count = int(top2[-1])
        top2_count = int(top2[-2])

        ent = _shannon_entropy_from_counts(counts)
        frac_pad = float(counts[pad_value]) / float(M) if M > 0 else 0.0

        feats[i, 0] = float(total_bytes)
        feats[i, 1] = float(top1_count)
        feats[i, 2] = float(top2_count)
        feats[i, 3] = float(ent)
        feats[i, 4] = float(frac_pad)

    return feats


# -----------------------------
# Data split for ResNet (PyTorch)
# -----------------------------
def split_training_and_test_resnet(ds, labels, train_indices_fold, test_indices_fold, M: int):
    """
    ds: (N, M_total) or (N, M) bytes
    Returns:
      X_train_t: (Ntr, 1, M) float32
      X_test_t : (Nte, 1, M) float32
      y_train  : (Ntr,) int64
      y_test   : (Nte,) int64
      X_train_raw_bytes: (Ntr, M) uint8 (for stats)
      X_test_raw_bytes : (Nte, M) uint8 (for stats)
    """
    X_train = ds[train_indices_fold]
    X_test  = ds[test_indices_fold]

    # ensure length M (truncate or pad)
    def _fix_len(X):
        if X.shape[1] == M:
            return X
        if X.shape[1] > M:
            return X[:, :M]
        # pad with 0
        pad = np.zeros((X.shape[0], M - X.shape[1]), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)

    X_train = _fix_len(X_train)
    X_test  = _fix_len(X_test)

    # keep raw bytes for stats (uint8)
    X_train_raw = X_train.astype(np.uint8, copy=False)
    X_test_raw  = X_test.astype(np.uint8, copy=False)

    # normalize to [0,1] and make (N,1,M)
    X_train_t = (X_train.astype(np.float32) / 255.0)[:, None, :]
    X_test_t  = (X_test.astype(np.float32) / 255.0)[:, None, :]

    y_train = labels[train_indices_fold].astype(np.int64)
    y_test  = labels[test_indices_fold].astype(np.int64)

    return X_train_t, X_test_t, y_train, y_test, X_train_raw, X_test_raw


# -----------------------------
# Train/Eval ResNet
# -----------------------------
def train_resnet(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 3,
    device: str = "cuda",
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if len(batch) == 2:
                xb, yb = batch
                sb = None
            else:
                xb, sb, yb = batch

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if sb is not None:
                sb = sb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb, stats=sb) if sb is not None else model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    xb, yb = batch
                    sb = None
                else:
                    xb, sb, yb = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if sb is not None:
                    sb = sb.to(device, non_blocking=True)

                logits = model(xb, stats=sb) if sb is not None else model(xb)
                loss = F.cross_entropy(logits, yb)
                bs = xb.size(0)
                val_loss += float(loss) * bs
                n += bs

        val_loss = val_loss / max(n, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate_resnet(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                xb, yb = batch
                sb = None
            else:
                xb, sb, yb = batch

            xb = xb.to(device, non_blocking=True)
            if sb is not None:
                sb = sb.to(device, non_blocking=True)

            logits = model(xb, stats=sb) if sb is not None else model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    return p, r, f1, y_pred


# -----------------------------
# One fold execution
# -----------------------------
def execute_fold_resnet(
    fold_idx: int,
    binary_numeric_labels: np.ndarray,
    scenario: int,
    param: int,
    train_idx,
    test_idx,
    M: int,
    use_stats: bool = False,
    stats_dim: int = 5,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout_p: float = 0.3,
    hidden_dim: int = 256,
    weight_decay: float = 0.0,
    device: str | None = None,
    debug_subset: bool = True,  # mimic your CNN "first+last 1000"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load dataset ----
    if param != 0:
        ds = np.load(f"datasets/re_bytes_{param}.npy")
        print(f"Experiment: ResNet classifier, fold {fold_idx}, scenario {scenario}, RE{param}, stats={use_stats}")
    else:
        ds = np.load("datasets/raw_bytes.npy")
        print(f"Experiment: ResNet classifier, fold {fold_idx}, scenario {scenario}, RAW, stats={use_stats}")

    # split -> normalize -> (N,1,M)
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = split_training_and_test_resnet(
        ds, binary_numeric_labels, np.array(train_idx), np.array(test_idx), M=M
    )

    # optional small subset (same spirit as your CNN demo)
    if debug_subset and len(X_train) > 2000:
        first = np.arange(0, 1000)
        last  = np.arange(len(X_train) - 1000, len(X_train))
        sel = np.concatenate([first, last])
        X_train = X_train[sel]
        y_train = y_train[sel]
        X_train_raw = X_train_raw[sel]

    # stats (if enabled)
    if use_stats:
        S_train = compute_stats_features(X_train_raw)  # (Ntr,5)
        S_test  = compute_stats_features(X_test_raw)   # (Nte,5)
        S_train_t = torch.from_numpy(S_train)
        S_test_t  = torch.from_numpy(S_test)
    else:
        S_train_t = None
        S_test_t  = None

    # tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    # train/val split
    n = len(X_train_t)
    perm = torch.randperm(n)
    val_n = max(1, int(0.1 * n))
    val_idx = perm[:val_n]
    tr_idx  = perm[val_n:]

    if use_stats:
        train_ds = TensorDataset(X_train_t[tr_idx], S_train_t[tr_idx], y_train_t[tr_idx])
        val_ds   = TensorDataset(X_train_t[val_idx], S_train_t[val_idx], y_train_t[val_idx])
        test_ds  = TensorDataset(X_test_t, S_test_t, y_test_t)
    else:
        train_ds = TensorDataset(X_train_t[tr_idx], y_train_t[tr_idx])
        val_ds   = TensorDataset(X_train_t[val_idx], y_train_t[val_idx])
        test_ds  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # model
    model = ResNet18_1D(
        in_channels=1,
        num_classes=2,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        use_stats=use_stats,
        stats_dim=stats_dim,
    ).to(device)

    train_resnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=3,
        device=device,
    )

    # eval
    p, r, f1, _ = evaluate_resnet(model=model, test_loader=test_loader, device=device)
    return p, r, f1


# -----------------------------
# Run ResNet for a scenario + representation
# -----------------------------
def run_resnet_for_scenario(
    scenario: int,
    prefix_for_files: str,   # "raw" or "re"
    global_label_encoder,
    keep_indices,
    param: int,
    M: int,
    use_stats: bool = False,
    stats_dim: int = 5,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout_p: float = 0.3,
    hidden_dim: int = 256,
    weight_decay: float = 0.0,
):
    labels = np.load(f"datasets/{prefix_for_files}_labels.npy")

    train_indices, test_indices = load_k_fold_results(
        f"k_fold_results/k_fold_s{scenario}_{prefix_for_files}.json"
    )

    if param != 0:
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels = np.where(numeric_labels == 0, 0, 1)

    k = len(train_indices)

    epochss = [15, 20, 25,30]
    batch_sizes = [256, 512, 1024]
    lrs = [1e-3, 5e-4, 1e-4]
    dropout_ps = [0.3, 0.5, 0.7]
    
    best_epoch = -1
    best_bs = -1
    best_lr = -1
    best_dps = -1 

    best_f1 = 0
    best_percision = 0
    best_recall = 0

    for e in epochss:
        for bs in batch_sizes:
            for learning_rate in lrs:
                for dp in dropout_ps:          
                    print(f'running with current parameters: epochs={e}, batch_size={bs}, learning_rate={learning_rate}, dropout_p={dp}')        
                    precisions, recalls, f1s = [], [], []
                    for fold_idx in range(k):
                        if len(train_indices[fold_idx]) == 0 or len(test_indices[fold_idx]) == 0:
                            continue

                        p, r, f1 = execute_fold_resnet(
                            fold_idx=fold_idx,
                            binary_numeric_labels=binary_numeric_labels,
                            scenario=scenario,
                            param=param,
                            train_idx=train_indices[fold_idx],
                            test_idx=test_indices[fold_idx],
                            M=M,
                            use_stats=use_stats,
                            stats_dim=stats_dim,
                            epochs=e,
                            batch_size=bs,
                            lr=learning_rate,
                            dropout_p=dp,
                            hidden_dim=hidden_dim,
                            weight_decay=weight_decay,
                        )

                        precisions.append(p)
                        recalls.append(r)
                        f1s.append(f1)
                        print(f"[Fold {fold_idx}] P={p:.4f} R={r:.4f} F1={f1:.4f}")
                    avg_f1 = float(np.mean(f1s))
                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        best_percision = float(np.mean(precisions))
                        best_recall = float(np.mean(recalls))
                        best_epoch = e
                        best_bs = bs
                        best_lr = learning_rate
                        best_dps = dp
    # if len(precisions) == 0:
    #     return 0.0, 0.0, 0.0
    print(f'best parameters are: {best_epoch}, {best_bs}, {best_lr}, {best_dps}')
    print(f'best results: P={best_percision:.4f} R={best_recall:.4f} F1={best_f1:.4f}')

    return best_percision, best_recall, best_f1  
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def run_resnet_classifier_on_dataset(
    scenario: int,
    prefix: str,               # "raw", "re5", "re10", "re15" (only used for naming)
    param: int,                # 0, 5, 10, 15
    global_label_encoder,
    M: int,
    use_stats: bool = False,
    out_csv: str = "results/resnet_summary.csv",
):
    if param != 0:
        keep_indices = get_keep_indices_from_fold0_ae(f"datasets/re_bytes_{param}.npy")
        prefix_for_files = "re"
    else:
        keep_indices = []
        prefix_for_files = "raw"

    avg_p, avg_r, avg_f1 = run_resnet_for_scenario(
        scenario=scenario,
        prefix_for_files=prefix_for_files,
        global_label_encoder=global_label_encoder,
        keep_indices=keep_indices,
        param=param,
        M=M,
        use_stats=use_stats,
        stats_dim=5,
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summary_path = Path(out_csv)
    write_header = not summary_path.exists()

    representation = "raw" if param == 0 else f"re{param}"

    with summary_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["scenario", "representation", "M", "use_stats", "avg_precision", "avg_recall", "avg_f1"])
        w.writerow([scenario, representation, M, int(use_stats), f"{avg_p:.6f}", f"{avg_r:.6f}", f"{avg_f1:.6f}"])

    return avg_p, avg_r, avg_f1


# -----------------------------
# The function you asked for:
# similar to run_experiment_cnn_classifier
# -----------------------------
def run_experiment_resnet_classifier(global_label_encoder, M_raw, M_re):
    """
    Runs ResNet classifier:
      - scenarios 2 and 3
      - representations raw, re5, re10, re15
      - runs BOTH variants:
          (A) without stats
          (B) with stats
    Appends to results/resnet_summary.csv
    """
    prefixes = ["raw", "re5", "re10", "re15"]
    params = [0, 5, 10, 15]

    results = {}
    for scenario in (2, 3):
        results[scenario] = {}
        for prefix, param in zip(prefixes, params):
            M = M_raw if param == 0 else M_re

            # no-stats
            key0 = f"{prefix}_nostats"
            results[scenario][key0] = run_resnet_classifier_on_dataset(
                scenario=scenario,
                prefix=prefix,
                param=param,
                global_label_encoder=global_label_encoder,
                M=M,
                use_stats=False,
                out_csv="results/resnet_summary.csv",
            )

            # with-stats
            key1 = f"{prefix}_withstats"
            results[scenario][key1] = run_resnet_classifier_on_dataset(
                scenario=scenario,
                prefix=prefix,
                param=param,
                global_label_encoder=global_label_encoder,
                M=M,
                use_stats=True,
                out_csv="results/resnet_summary.csv",
            )

    return results
