"""
Final Module 2 pipeline for CH-Regime-220-v1

PI-MambaRegime: Physics-Informed Mamba-style temporal representation learning
for hidden dynamical regime discovery.

Run from project root:
    python scripts/run_module2_final.py --epochs 25

Expected input:
    data/frozen/CH-Regime-220-v1/ch_regime_physics_trajectories.csv

Outputs:
    results/module2_final/windows
    results/module2_final/models
    results/module2_final/embeddings
    results/module2_final/regimes
    results/module2_final/figures
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "frozen" / "CH-Regime-220-v1" / "ch_regime_physics_trajectories.csv"

OUT_ROOT = ROOT / "results" / "module2_final"
WINDOW_DIR = OUT_ROOT / "windows"
MODEL_DIR = OUT_ROOT / "models"
EMBED_DIR = OUT_ROOT / "embeddings"
REGIME_DIR = OUT_ROOT / "regimes"
FIG_DIR = OUT_ROOT / "figures"

for d in [WINDOW_DIR, MODEL_DIR, EMBED_DIR, REGIME_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "T_K",
    "Tj_in_K",
    "Tj_out_K",
    "valve_actual_pct",
    "feed_rate_kg_s",
    "mM_kg",
    "mP_kg",
    "mM_added_kg",
    "conversion_proxy",
    "solids_fraction",
    "viscosity_proxy",
    "U_kW_m2_K",
    "reaction_rate_kg_s",
    "Q_feed_kW",
    "Q_reaction_kW",
    "Q_jacket_kW",
    "Q_loss_kW",
    "dTdt_K_min",
]

META_COLS = [
    "trajectory_id",
    "time_s",
    "time_min",
    "dataset_split",
    "recipe_stage",
    "product",
    "batch_number",
    "season",
    "disturbance_type",
]

PHYSICS_INTERPRET_COLS = [
    "T_K",
    "dTdt_K_min",
    "feed_rate_kg_s",
    "viscosity_proxy",
    "U_kW_m2_K",
    "reaction_rate_kg_s",
    "Q_feed_kW",
    "Q_reaction_kW",
    "Q_jacket_kW",
    "Q_loss_kW",
    "conversion_proxy",
    "solids_fraction",
    "valve_actual_pct",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def assert_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


# -----------------------------------------------------------------------------
# Window builder
# -----------------------------------------------------------------------------

def build_windows(window_len: int, stride: int, pred_horizon: int, rebuild: bool = False) -> None:
    x_path = WINDOW_DIR / "X_windows.npy"
    y_future_path = WINDOW_DIR / "y_future.npy"
    y_recon_path = WINDOW_DIR / "y_recon.npy"
    meta_path = WINDOW_DIR / "window_metadata.csv"

    if x_path.exists() and y_future_path.exists() and y_recon_path.exists() and meta_path.exists() and not rebuild:
        print("Window files already exist. Use --rebuild_windows to rebuild.")
        return

    print("\n[1/5] Building physics-aware sliding windows")
    print(f"Loading frozen dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    assert_columns(df, FEATURE_COLS + META_COLS)

    print(f"Dataset shape: {df.shape}")
    print(f"Trajectories: {df['trajectory_id'].nunique()}")

    train_df = df[df["dataset_split"] == "train"].copy()
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))

    scaled = scaler.transform(df[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)

    X_list = []
    y_future_list = []
    y_recon_list = []
    meta_rows = []

    for tid, g in tqdm(df.groupby("trajectory_id", sort=True), desc="Trajectories"):
        g = g.reset_index(drop=False)
        source_indices = g["index"].to_numpy()
        sg = scaled[source_indices]

        n = len(g)
        max_start = n - window_len - pred_horizon
        if max_start <= 0:
            continue

        for start in range(0, max_start, stride):
            end = start + window_len
            center = end - 1
            future = end + pred_horizon - 1

            X_list.append(sg[start:end])
            y_recon_list.append(sg[center])
            y_future_list.append(sg[future])

            center_row = g.iloc[center]
            future_row = g.iloc[future]

            meta = {c: center_row[c] for c in META_COLS}
            meta["source_row_index"] = int(center_row["index"])
            meta["window_start_time_min"] = float(g.iloc[start]["time_min"])
            meta["window_end_time_min"] = float(center_row["time_min"])
            meta["future_time_min"] = float(future_row["time_min"])

            for c in PHYSICS_INTERPRET_COLS:
                meta[c] = center_row[c]

            X_list.append if False else None
            meta_rows.append(meta)

    X = np.stack(X_list).astype(np.float32)
    y_future = np.stack(y_future_list).astype(np.float32)
    y_recon = np.stack(y_recon_list).astype(np.float32)
    meta = pd.DataFrame(meta_rows)

    print(f"X windows: {X.shape}")
    print(f"Future targets: {y_future.shape}")
    print(f"Recon targets: {y_recon.shape}")
    print(f"Metadata: {meta.shape}")

    np.save(x_path, X)
    np.save(y_future_path, y_future)
    np.save(y_recon_path, y_recon)
    meta.to_csv(meta_path, index=False)
    joblib.dump(scaler, WINDOW_DIR / "feature_scaler.pkl")

    raw = train_df[["reaction_rate_kg_s", "Q_reaction_kW"]].copy()
    raw = raw[raw["reaction_rate_kg_s"] > 1e-10]
    q_slope = float(np.median(raw["Q_reaction_kW"] / raw["reaction_rate_kg_s"]))

    config = {
        "dataset_version": "CH-Regime-220-v1",
        "data_path": str(DATA_PATH),
        "window_len": window_len,
        "stride": stride,
        "pred_horizon": pred_horizon,
        "feature_cols": FEATURE_COLS,
        "meta_cols": META_COLS,
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[2]),
        "physics_constants": {
            "q_reaction_per_reaction_rate": q_slope,
            "U_lower_bound": 0.2,
        },
    }
    save_json(config, WINDOW_DIR / "window_config.json")

    print("Saved windows to:", WINDOW_DIR)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y_future: np.ndarray, y_recon: np.ndarray, indices: np.ndarray):
        self.X = X[indices]
        self.y_future = y_future[indices]
        self.y_recon = y_recon[indices]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_future[idx], dtype=torch.float32),
            torch.tensor(self.y_recon[idx], dtype=torch.float32),
        )


class XOnlyDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx], dtype=torch.float32)


# -----------------------------------------------------------------------------
# Final PI-MambaRegime model
# -----------------------------------------------------------------------------

class SelectiveSSMBlock(nn.Module):
    """
    CPU/Windows-friendly Mamba-inspired selective state-space block.

    This is a custom selective SSM layer for this research pipeline, not the
    CUDA mamba-ssm package. It keeps the core idea: input-dependent selection,
    gated state updates, and long-range temporal representation learning.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.select_proj = nn.Linear(d_model, 2 * d_model)
        self.A_logit = nn.Parameter(torch.randn(d_model) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        u_gate = self.in_proj(x)
        u, gate = u_gate.chunk(2, dim=-1)
        u = F.silu(u)
        gate = torch.sigmoid(gate)

        select = self.select_proj(x)
        b_raw, c_raw = select.chunk(2, dim=-1)
        b_t = torch.sigmoid(b_raw)
        c_t = torch.sigmoid(c_raw)

        A = torch.sigmoid(self.A_logit).view(1, -1)
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            h = A * h + b_t[:, t, :] * u[:, t, :]
            y = c_t[:, t, :] * h
            ys.append(y.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        y = y * gate
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class PIMambaRegime(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 3,
        embedding_dim: int = 24,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(d_model, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.embedding_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_features),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_features),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        pooled = h[:, -1, :]
        z = self.embedding_head(pooled)
        z = F.normalize(z, dim=-1)
        return {
            "embedding": z,
            "pred_future": self.pred_head(z),
            "recon_last": self.recon_head(z),
        }


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def inverse_scale_torch(x_norm: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_norm * scale + mean


def embedding_variance_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-6)
    return torch.mean(F.relu(0.5 - std))


def physics_prediction_loss(
    pred_norm: torch.Tensor,
    feature_cols: List[str],
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    physics_constants: Dict[str, float],
) -> torch.Tensor:
    pred_raw = inverse_scale_torch(pred_norm, scaler_mean, scaler_scale)

    idx_rate = feature_cols.index("reaction_rate_kg_s")
    idx_qrxn = feature_cols.index("Q_reaction_kW")
    idx_u = feature_cols.index("U_kW_m2_K")
    idx_feed = feature_cols.index("feed_rate_kg_s")

    rate = pred_raw[:, idx_rate]
    qrxn = pred_raw[:, idx_qrxn]
    U = pred_raw[:, idx_u]
    feed = pred_raw[:, idx_feed]

    q_slope = float(physics_constants["q_reaction_per_reaction_rate"])
    U_lower = float(physics_constants["U_lower_bound"])

    q_scale = scaler_scale[idx_qrxn]
    u_scale = scaler_scale[idx_u]
    feed_scale = scaler_scale[idx_feed]

    q_loss = torch.mean(((qrxn - q_slope * rate) / (q_scale + 1e-8)) ** 2)
    u_bound_loss = torch.mean((F.relu(U_lower - U) / (u_scale + 1e-8)) ** 2)
    feed_bound_loss = torch.mean((F.relu(-feed) / (feed_scale + 1e-8)) ** 2)

    return q_loss + u_bound_loss + feed_bound_loss


def compute_loss(
    out: Dict[str, torch.Tensor],
    y_future: torch.Tensor,
    y_recon: torch.Tensor,
    feature_cols: List[str],
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    physics_constants: Dict[str, float],
    lambda_recon: float,
    lambda_physics: float,
    lambda_variance: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_loss = F.mse_loss(out["pred_future"], y_future)
    recon_loss = F.mse_loss(out["recon_last"], y_recon)
    phys_loss = physics_prediction_loss(
        out["pred_future"], feature_cols, scaler_mean, scaler_scale, physics_constants
    )
    var_loss = embedding_variance_loss(out["embedding"])

    loss = pred_loss + lambda_recon * recon_loss + lambda_physics * phys_loss + lambda_variance * var_loss

    logs = {
        "loss": float(loss.detach().cpu()),
        "pred_loss": float(pred_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "physics_loss": float(phys_loss.detach().cpu()),
        "variance_loss": float(var_loss.detach().cpu()),
    }
    return loss, logs


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_model(args) -> None:
    print("\n[2/5] Training final PI-MambaRegime model")

    X = np.load(WINDOW_DIR / "X_windows.npy")
    y_future = np.load(WINDOW_DIR / "y_future.npy")
    y_recon = np.load(WINDOW_DIR / "y_recon.npy")
    meta = pd.read_csv(WINDOW_DIR / "window_metadata.csv")
    scaler = joblib.load(WINDOW_DIR / "feature_scaler.pkl")

    with open(WINDOW_DIR / "window_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    feature_cols = config["feature_cols"]
    physics_constants = config["physics_constants"]
    n_features = len(feature_cols)

    train_idx = meta.index[meta["dataset_split"] == "train"].to_numpy()
    val_idx = meta.index[meta["dataset_split"] == "val"].to_numpy()

    train_ds = WindowDataset(X, y_future, y_recon, train_idx)
    val_ds = WindowDataset(X, y_future, y_recon, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")

    model = PIMambaRegime(
        n_features=n_features,
        d_model=args.d_model,
        n_layers=args.n_layers,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "pred_loss": 0.0, "recon_loss": 0.0, "physics_loss": 0.0, "variance_loss": 0.0}
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for Xb, yf, yr in pbar:
            Xb = Xb.to(device)
            yf = yf.to(device)
            yr = yr.to(device)

            optimizer.zero_grad()
            out = model(Xb)
            loss, logs = compute_loss(
                out,
                yf,
                yr,
                feature_cols,
                scaler_mean,
                scaler_scale,
                physics_constants,
                args.lambda_recon,
                args.lambda_physics,
                args.lambda_variance,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = Xb.size(0)
            for k in sums:
                sums[k] += logs[k] * bs
            n_seen += bs
            pbar.set_postfix({"loss": sums["loss"] / max(n_seen, 1), "pred": logs["pred_loss"], "phy": logs["physics_loss"]})

        scheduler.step()
        train_logs = {k: sums[k] / max(n_seen, 1) for k in sums}

        val_logs = evaluate_model(
            model,
            val_loader,
            device,
            feature_cols,
            scaler_mean,
            scaler_scale,
            physics_constants,
            args,
        )

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}, **{f"val_{k}": v for k, v in val_logs.items()}}
        history.append(row)

        print(
            f"Epoch {epoch}: train_loss={train_logs['loss']:.6f}, "
            f"val_loss={val_logs['loss']:.6f}, val_pred={val_logs['pred_loss']:.6f}, "
            f"val_physics={val_logs['physics_loss']:.6f}"
        )

        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            best_epoch = epoch
            patience_left = args.patience
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_features": n_features,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "embedding_dim": args.embedding_dim,
                    "dropout": args.dropout,
                    "feature_cols": feature_cols,
                },
                "training_config": vars(args),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, MODEL_DIR / "pi_mamba_regime_final_best.pt")
            print(f"Saved best model at epoch {epoch}: val_loss={best_val:.6f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    pd.DataFrame(history).to_csv(MODEL_DIR / "training_history.csv", index=False)

    plot_training_history(MODEL_DIR / "training_history.csv")

    print("Training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val:.6f}")


def evaluate_model(
    model,
    loader,
    device,
    feature_cols,
    scaler_mean,
    scaler_scale,
    physics_constants,
    args,
) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "pred_loss": 0.0, "recon_loss": 0.0, "physics_loss": 0.0, "variance_loss": 0.0}
    n_seen = 0

    with torch.no_grad():
        for Xb, yf, yr in loader:
            Xb = Xb.to(device)
            yf = yf.to(device)
            yr = yr.to(device)

            out = model(Xb)
            loss, logs = compute_loss(
                out,
                yf,
                yr,
                feature_cols,
                scaler_mean,
                scaler_scale,
                physics_constants,
                args.lambda_recon,
                args.lambda_physics,
                args.lambda_variance,
            )
            bs = Xb.size(0)
            for k in sums:
                sums[k] += logs[k] * bs
            n_seen += bs

    return {k: sums[k] / max(n_seen, 1) for k in sums}


def plot_training_history(history_path: Path) -> None:
    hist = pd.read_csv(history_path)
    plt.figure(figsize=(8, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train total loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PI-MambaRegime Training History")
    plt.legend()
    savefig("01_training_history_total_loss.png")

    plt.figure(figsize=(8, 4))
    plt.plot(hist["epoch"], hist["val_pred_loss"], label="Validation future prediction loss")
    plt.plot(hist["epoch"], hist["val_recon_loss"], label="Validation reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Prediction and Reconstruction Loss")
    plt.legend()
    savefig("02_training_history_component_losses.png")


# -----------------------------------------------------------------------------
# Embedding extraction
# -----------------------------------------------------------------------------

def extract_embeddings(batch_size: int) -> None:
    print("\n[3/5] Extracting final PI-Mamba embeddings")

    X = np.load(WINDOW_DIR / "X_windows.npy")
    meta = pd.read_csv(WINDOW_DIR / "window_metadata.csv")

    checkpoint = torch.load(MODEL_DIR / "pi_mamba_regime_final_best.pt", map_location="cpu")
    cfg = checkpoint["model_config"]

    model = PIMambaRegime(
        n_features=cfg["n_features"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        embedding_dim=cfg["embedding_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(XOnlyDataset(X), batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings = []
    pred_future = []
    recon_last = []

    with torch.no_grad():
        for Xb in tqdm(loader, desc="Embeddings"):
            Xb = Xb.to(device)
            out = model(Xb)
            embeddings.append(out["embedding"].cpu().numpy())
            pred_future.append(out["pred_future"].cpu().numpy())
            recon_last.append(out["recon_last"].cpu().numpy())

    Z = np.concatenate(embeddings, axis=0)
    pred = np.concatenate(pred_future, axis=0)
    recon = np.concatenate(recon_last, axis=0)

    np.save(EMBED_DIR / "pi_mamba_embeddings.npy", Z)
    np.save(EMBED_DIR / "pred_future_norm.npy", pred)
    np.save(EMBED_DIR / "recon_last_norm.npy", recon)
    meta.to_csv(EMBED_DIR / "embedding_metadata.csv", index=False)

    print(f"Saved embeddings: {Z.shape} -> {EMBED_DIR / 'pi_mamba_embeddings.npy'}")


# -----------------------------------------------------------------------------
# Regime discovery + reality test
# -----------------------------------------------------------------------------

def cluster_regimes(k_min: int, k_max: int, seed: int) -> None:
    print("\n[4/5] Discovering hidden regimes")

    Z = np.load(EMBED_DIR / "pi_mamba_embeddings.npy")
    meta = pd.read_csv(EMBED_DIR / "embedding_metadata.csv")

    emb_scaler = StandardScaler()
    Zs = emb_scaler.fit_transform(Z)
    joblib.dump(emb_scaler, REGIME_DIR / "embedding_scaler.pkl")

    pca_dim = min(16, Zs.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    Zp = pca.fit_transform(Zs)
    joblib.dump(pca, REGIME_DIR / "embedding_pca.pkl")

    train_mask = meta["dataset_split"].values == "train"
    Z_train = Zp[train_mask]

    ks = list(range(k_min, k_max + 1))
    score_rows = []
    models = {}

    rng = np.random.default_rng(seed)
    metric_idx = np.arange(len(Z_train))
    if len(metric_idx) > 10000:
        metric_idx = rng.choice(metric_idx, size=10000, replace=False)

    for k in ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            n_init=5,
            reg_covar=1e-5,
        )
        train_labels = gmm.fit_predict(Z_train)
        metric_labels = train_labels[metric_idx]
        metric_Z = Z_train[metric_idx]

        sil = silhouette_score(metric_Z, metric_labels)
        db = davies_bouldin_score(metric_Z, metric_labels)
        bic = gmm.bic(Z_train)
        aic = gmm.aic(Z_train)

        score_rows.append({"K": k, "silhouette": sil, "davies_bouldin": db, "BIC": bic, "AIC": aic})
        models[k] = gmm
        print(f"K={k}: silhouette={sil:.4f}, DB={db:.4f}, BIC={bic:.2f}, AIC={aic:.2f}")

    score_df = pd.DataFrame(score_rows)
    score_df["rank_silhouette"] = score_df["silhouette"].rank(ascending=False)
    score_df["rank_db"] = score_df["davies_bouldin"].rank(ascending=True)
    score_df["rank_bic"] = score_df["BIC"].rank(ascending=True)
    score_df["combined_rank"] = score_df["rank_silhouette"] + score_df["rank_db"] + 0.25 * score_df["rank_bic"]
    best_k = int(score_df.sort_values("combined_rank").iloc[0]["K"])
    best_model = models[best_k]

    score_df.to_csv(REGIME_DIR / "cluster_model_selection.csv", index=False)
    joblib.dump(best_model, REGIME_DIR / "gmm_regime_model.pkl")

    labels = best_model.predict(Zp)
    probs = best_model.predict_proba(Zp).max(axis=1)

    assign = meta.copy()
    assign["regime_id"] = labels.astype(int)
    assign["regime_probability"] = probs.astype(float)

    assign.to_csv(REGIME_DIR / "regime_assignments.csv", index=False)

    print(f"Selected K: {best_k}")
    print(f"Saved assignments: {REGIME_DIR / 'regime_assignments.csv'}")

    summary = make_regime_summary(assign)
    coherence = compute_temporal_coherence(assign)
    separability = compute_physics_separability(assign)
    transition = compute_transition_matrix(assign)

    status = write_regime_reality_report(assign, score_df, summary, coherence, separability, transition, best_k)

    plot_cluster_selection(score_df)
    plot_latent_pca(Zp, assign)
    plot_regime_timelines(assign)
    plot_regime_boxplots(assign)
    plot_transition_matrix(transition)

    print("\nRegime discovery complete.")
    print(f"Reality Test Status: {status}")


def make_regime_summary(assign: pd.DataFrame) -> pd.DataFrame:
    summary = (
        assign.groupby("regime_id")
        .agg(
            n_windows=("regime_id", "size"),
            n_trajectories=("trajectory_id", "nunique"),
            dominant_stage=("recipe_stage", lambda x: x.value_counts().index[0]),
            dominant_split=("dataset_split", lambda x: x.value_counts().index[0]),
            dominant_product=("product", lambda x: x.value_counts().index[0]),
            mean_T_K=("T_K", "mean"),
            mean_dTdt_K_min=("dTdt_K_min", "mean"),
            mean_feed_rate=("feed_rate_kg_s", "mean"),
            mean_viscosity=("viscosity_proxy", "mean"),
            mean_U=("U_kW_m2_K", "mean"),
            mean_Q_reaction=("Q_reaction_kW", "mean"),
            mean_Q_jacket=("Q_jacket_kW", "mean"),
            mean_conversion=("conversion_proxy", "mean"),
            mean_valve=("valve_actual_pct", "mean"),
            mean_probability=("regime_probability", "mean"),
        )
        .reset_index()
    )
    summary["window_pct"] = 100.0 * summary["n_windows"] / len(assign)
    summary.to_csv(REGIME_DIR / "regime_summary.csv", index=False)
    return summary


def compute_temporal_coherence(assign: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, g in assign.groupby("trajectory_id"):
        g = g.sort_values("time_s")
        labels = g["regime_id"].to_numpy()
        if len(labels) == 0:
            continue
        current = labels[0]
        run_len = 1
        for r in labels[1:]:
            if r == current:
                run_len += 1
            else:
                rows.append({"trajectory_id": tid, "regime_id": current, "run_length_windows": run_len})
                current = r
                run_len = 1
        rows.append({"trajectory_id": tid, "regime_id": current, "run_length_windows": run_len})

    runs = pd.DataFrame(rows)
    runs.to_csv(REGIME_DIR / "regime_temporal_runs.csv", index=False)

    coherence = (
        runs.groupby("regime_id")
        .agg(
            median_run_length_windows=("run_length_windows", "median"),
            mean_run_length_windows=("run_length_windows", "mean"),
            max_run_length_windows=("run_length_windows", "max"),
        )
        .reset_index()
    )
    coherence.to_csv(REGIME_DIR / "regime_temporal_coherence.csv", index=False)
    return coherence


def compute_physics_separability(assign: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in PHYSICS_INTERPRET_COLS:
        overall_mean = assign[col].mean()
        between = 0.0
        within = 0.0
        for rid, g in assign.groupby("regime_id"):
            n = len(g)
            between += n * (g[col].mean() - overall_mean) ** 2
            within += ((g[col] - g[col].mean()) ** 2).sum()
        rows.append({"feature": col, "between_within_ratio": between / (within + 1e-12)})
    sep = pd.DataFrame(rows).sort_values("between_within_ratio", ascending=False)
    sep.to_csv(REGIME_DIR / "regime_physics_separability.csv", index=False)
    return sep


def compute_transition_matrix(assign: pd.DataFrame) -> pd.DataFrame:
    regimes = sorted(assign["regime_id"].unique())
    idx = {r: i for i, r in enumerate(regimes)}
    mat = np.zeros((len(regimes), len(regimes)), dtype=float)

    for tid, g in assign.groupby("trajectory_id"):
        labels = g.sort_values("time_s")["regime_id"].to_numpy()
        for a, b in zip(labels[:-1], labels[1:]):
            mat[idx[a], idx[b]] += 1.0

    row_sums = mat.sum(axis=1, keepdims=True)
    prob = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    df = pd.DataFrame(prob, index=[f"R{r}" for r in regimes], columns=[f"R{r}" for r in regimes])
    df.to_csv(REGIME_DIR / "regime_transition_matrix.csv")
    return df


def write_regime_reality_report(
    assign: pd.DataFrame,
    score_df: pd.DataFrame,
    summary: pd.DataFrame,
    coherence: pd.DataFrame,
    separability: pd.DataFrame,
    transition: pd.DataFrame,
    best_k: int,
) -> str:
    nmi_stage = normalized_mutual_info_score(assign["recipe_stage"], assign["regime_id"])
    nmi_product = normalized_mutual_info_score(assign["product"], assign["regime_id"])
    nmi_season = normalized_mutual_info_score(assign["season"], assign["regime_id"])
    nmi_split = normalized_mutual_info_score(assign["dataset_split"], assign["regime_id"])

    min_traj = int(summary["n_trajectories"].min())
    min_pct = float(summary["window_pct"].min())
    median_run = float(coherence["median_run_length_windows"].median())
    top_sep = float(separability["between_within_ratio"].head(5).mean())
    mean_prob = float(summary["mean_probability"].mean())

    gates = {
        "regime_count_reasonable_4_to_8": 4 <= best_k <= 8,
        "all_regimes_recurrent_min_10_trajectories": min_traj >= 10,
        "no_tiny_regime_min_1_percent": min_pct >= 1.0,
        "temporal_coherence_median_run_ge_2_windows": median_run >= 2.0,
        "physics_separability_top5_mean_gt_0p05": top_sep > 0.05,
        "not_identical_to_recipe_stage_NMI_lt_0p90": nmi_stage < 0.90,
        "not_identical_to_product_NMI_lt_0p80": nmi_product < 0.80,
        "average_assignment_probability_gt_0p50": mean_prob > 0.50,
    }
    status = "PASS" if all(gates.values()) else "NEEDS_ATTENTION"

    path = REGIME_DIR / "regime_reality_test_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("MODULE 2 FINAL REGIME REALITY TEST REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write("METHOD\n")
        f.write("-" * 72 + "\n")
        f.write("PI-MambaRegime: physics-informed Mamba-style temporal representation learning\n")
        f.write("Training labels used for regime discovery: NONE\n")
        f.write("Known labels used only for post-hoc interpretation: recipe_stage, product, season, split\n\n")

        f.write(f"Selected K: {best_k}\n")
        f.write(f"Overall status: {status}\n\n")

        f.write("CLUSTER MODEL SELECTION\n")
        f.write("-" * 72 + "\n")
        f.write(score_df.to_string(index=False))
        f.write("\n\n")

        f.write("REGIME SUMMARY\n")
        f.write("-" * 72 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")

        f.write("TEMPORAL COHERENCE\n")
        f.write("-" * 72 + "\n")
        f.write(coherence.to_string(index=False))
        f.write("\n\n")

        f.write("PHYSICS SEPARABILITY TOP FEATURES\n")
        f.write("-" * 72 + "\n")
        f.write(separability.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("LABEL-INDEPENDENCE CHECK\n")
        f.write("-" * 72 + "\n")
        f.write(f"NMI(regime, recipe_stage): {nmi_stage:.4f}\n")
        f.write(f"NMI(regime, product): {nmi_product:.4f}\n")
        f.write(f"NMI(regime, season): {nmi_season:.4f}\n")
        f.write(f"NMI(regime, dataset_split): {nmi_split:.4f}\n\n")

        f.write("ACCEPTANCE GATES\n")
        f.write("-" * 72 + "\n")
        for k, v in gates.items():
            f.write(f"{'PASS' if v else 'CHECK'}: {k}\n")

        f.write("\nFINAL STATUS\n")
        f.write("-" * 72 + "\n")
        f.write(status + "\n")

    print(f"Saved report: {path}")
    return status


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_cluster_selection(score_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(score_df["K"], score_df["silhouette"], marker="o", label="Silhouette")
    plt.xlabel("Number of regimes K")
    plt.ylabel("Silhouette score")
    plt.title("Cluster Model Selection: Silhouette")
    plt.legend()
    savefig("03_cluster_selection_silhouette.png")

    plt.figure(figsize=(8, 4))
    plt.plot(score_df["K"], score_df["davies_bouldin"], marker="o", label="Davies-Bouldin")
    plt.xlabel("Number of regimes K")
    plt.ylabel("Davies-Bouldin score")
    plt.title("Cluster Model Selection: Davies-Bouldin")
    plt.legend()
    savefig("04_cluster_selection_davies_bouldin.png")


def plot_latent_pca(Zp: np.ndarray, assign: pd.DataFrame) -> None:
    Z2 = PCA(n_components=2, random_state=42).fit_transform(Zp)

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=assign["regime_id"], s=4, alpha=0.45)
    plt.colorbar(sc, label="Discovered regime")
    plt.xlabel("Latent PC1")
    plt.ylabel("Latent PC2")
    plt.title("PI-MambaRegime Latent Space by Discovered Regime")
    savefig("05_latent_pca_by_regime.png")

    stage_names = sorted(assign["recipe_stage"].unique())
    stage_map = {s: i for i, s in enumerate(stage_names)}
    stage_code = assign["recipe_stage"].map(stage_map)
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=stage_code, s=4, alpha=0.45)
    plt.colorbar(sc, label=str(stage_map))
    plt.xlabel("Latent PC1")
    plt.ylabel("Latent PC2")
    plt.title("PI-MambaRegime Latent Space by Known Recipe Stage")
    savefig("06_latent_pca_by_recipe_stage.png")


def plot_regime_timelines(assign: pd.DataFrame) -> None:
    sample_ids = []
    for split in ["baseline", "train", "val", "test", "ood"]:
        ids = assign.loc[assign["dataset_split"] == split, "trajectory_id"].drop_duplicates().head(2).tolist()
        sample_ids.extend(ids)

    for tid in sample_ids:
        d = assign[assign["trajectory_id"] == tid].sort_values("time_min")
        plt.figure(figsize=(11, 3))
        plt.scatter(d["time_min"], d["regime_id"], c=d["regime_id"], s=9)
        plt.xlabel("Time (min)")
        plt.ylabel("Regime ID")
        plt.title(
            f"Regime Timeline | trajectory={tid} | split={d['dataset_split'].iloc[0]} | "
            f"product={d['product'].iloc[0]} | batch={d['batch_number'].iloc[0]} | season={d['season'].iloc[0]}"
        )
        savefig(f"07_regime_timeline_tid_{tid}.png")


def plot_regime_boxplots(assign: pd.DataFrame) -> None:
    features = [
        "Q_reaction_kW",
        "Q_jacket_kW",
        "U_kW_m2_K",
        "viscosity_proxy",
        "feed_rate_kg_s",
        "dTdt_K_min",
        "conversion_proxy",
    ]
    regimes = sorted(assign["regime_id"].unique())
    for col in features:
        data = [assign.loc[assign["regime_id"] == r, col].values for r in regimes]
        plt.figure(figsize=(8, 4))
        plt.boxplot(data, tick_labels=[str(r) for r in regimes], showfliers=False)
        plt.xlabel("Discovered regime ID")
        plt.ylabel(col)
        plt.title(f"Physics Feature Distribution by Regime: {col}")
        savefig(f"08_regime_boxplot_{col}.png")


def plot_transition_matrix(transition: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(transition.values, aspect="auto")
    plt.colorbar(label="Transition probability")
    plt.xticks(range(len(transition.columns)), transition.columns, rotation=45)
    plt.yticks(range(len(transition.index)), transition.index)
    plt.xlabel("Next regime")
    plt.ylabel("Current regime")
    plt.title("Discovered Regime Transition Matrix")
    savefig("09_regime_transition_matrix.png")


# -----------------------------------------------------------------------------
# Final manifest
# -----------------------------------------------------------------------------

def write_manifest(args) -> None:
    print("\n[5/5] Writing final Module 2 manifest")
    manifest = {
        "module": "Module 2",
        "method_name": "PI-MambaRegime",
        "dataset": "CH-Regime-220-v1",
        "input_dataset": str(DATA_PATH),
        "output_root": str(OUT_ROOT),
        "window_dir": str(WINDOW_DIR),
        "model_dir": str(MODEL_DIR),
        "embedding_dir": str(EMBED_DIR),
        "regime_dir": str(REGIME_DIR),
        "figure_dir": str(FIG_DIR),
        "command_config": vars(args),
        "important_files": {
            "best_model": str(MODEL_DIR / "pi_mamba_regime_final_best.pt"),
            "training_history": str(MODEL_DIR / "training_history.csv"),
            "embeddings": str(EMBED_DIR / "pi_mamba_embeddings.npy"),
            "regime_assignments": str(REGIME_DIR / "regime_assignments.csv"),
            "regime_summary": str(REGIME_DIR / "regime_summary.csv"),
            "regime_reality_test": str(REGIME_DIR / "regime_reality_test_report.txt"),
        },
    }
    save_json(manifest, OUT_ROOT / "MODULE2_FINAL_MANIFEST.json")
    print(f"Saved manifest: {OUT_ROOT / 'MODULE2_FINAL_MANIFEST.json'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Final Module 2 PI-MambaRegime pipeline")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)

    parser.add_argument("--window_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--pred_horizon", type=int, default=8)
    parser.add_argument("--rebuild_windows", action="store_true")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--lambda_recon", type=float, default=0.30)
    parser.add_argument("--lambda_physics", type=float, default=0.010)
    parser.add_argument("--lambda_variance", type=float, default=0.001)

    parser.add_argument("--k_min", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Frozen dataset not found: {DATA_PATH}\n"
            "Make sure Module 1 is frozen as data/frozen/CH-Regime-220-v1."
        )

    print("=" * 80)
    print("FINAL MODULE 2 PIPELINE: PI-MambaRegime")
    print("=" * 80)
    print(f"Frozen dataset: {DATA_PATH}")
    print(f"Output root: {OUT_ROOT}")

    build_windows(args.window_len, args.stride, args.pred_horizon, rebuild=args.rebuild_windows)
    train_model(args)
    extract_embeddings(args.embedding_batch_size)
    cluster_regimes(args.k_min, args.k_max, args.seed)
    write_manifest(args)

    print("\n" + "=" * 80)
    print("MODULE 2 FINAL PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Reality report: {REGIME_DIR / 'regime_reality_test_report.txt'}")
    print(f"Regime assignments: {REGIME_DIR / 'regime_assignments.csv'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
