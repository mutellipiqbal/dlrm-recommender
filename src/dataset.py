"""
Synthetic Dataset — src/dataset.py
====================================
Generates a Criteo-style dataset with both dense and sparse features.
This mirrors what the Databricks DLRM reference notebook does.

Feature schema (mimics real-world re-ranking signals):
    Dense  (13 features): user_age_norm, item_price_norm, days_active_norm,
                          avg_session_len_norm, click_rate_norm,
                          item_popularity_norm, recency_norm,
                          price_sensitivity_norm, brand_affinity_norm,
                          category_affinity_norm, seasonal_score_norm,
                          social_score_norm, device_score_norm
    Sparse (6 features) : user_id, item_id, category_id, brand_id,
                          country_id, device_type_id
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ─── Vocab sizes (mimics real categorical cardinalities) ──────────────────────
VOCAB_SIZES = {
    "user_id":       100_000,
    "item_id":        50_000,
    "category_id":       500,
    "brand_id":        5_000,
    "country_id":        200,
    "device_type_id":      5,
}

NUM_DENSE   = 13
NUM_SPARSE  = len(VOCAB_SIZES)
SPARSE_KEYS = list(VOCAB_SIZES.keys())


def generate_synthetic_data(
    n_samples: int = 500_000,
    seed: int = 42,
    pos_rate: float = 0.15,   # realistic CTR / purchase rate
) -> pd.DataFrame:
    """
    Generate synthetic Criteo-style interaction data.
    Labels have a signal: items from the user's preferred category
    are more likely to be positive.
    """
    rng = np.random.default_rng(seed)

    # ── Sparse features ──────────────────────────────────────────────────────
    sparse_data: dict[str, np.ndarray] = {}
    for col, vocab in VOCAB_SIZES.items():
        sparse_data[col] = rng.integers(1, vocab + 1, size=n_samples)   # 1-based, 0=padding

    # ── Dense features (13, like Criteo) ─────────────────────────────────────
    dense_data = rng.standard_normal((n_samples, NUM_DENSE)).astype(np.float32)
    dense_data = np.clip(dense_data, -5, 5)

    # ── Labels with realistic signal ─────────────────────────────────────────
    # Users prefer certain categories → add signal so model can learn
    user_preferred_cat = sparse_data["user_id"] % VOCAB_SIZES["category_id"] + 1
    item_cat = sparse_data["category_id"]
    match_signal = (user_preferred_cat == item_cat).astype(np.float32)

    # Price sensitivity: cheaper items more likely to be clicked
    price_signal = (dense_data[:, 1] < 0).astype(np.float32)   # dense[:,1] ≈ price_norm

    # Combine: base rate + signal + noise
    logit = (
        -2.0                                # baseline → ~12% positive rate
        + 1.5 * match_signal
        + 0.8 * price_signal
        + 0.3 * rng.standard_normal(n_samples)
    )
    prob   = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.uniform(size=n_samples) < prob).astype(np.float32)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    dense_cols = {f"dense_{i}": dense_data[:, i] for i in range(NUM_DENSE)}
    df = pd.DataFrame({**dense_cols, **sparse_data, "label": labels})

    pos_rate_actual = labels.mean()
    print(
        f"Generated {n_samples:,} samples | "
        f"Positive rate: {pos_rate_actual:.2%} | "
        f"Features: {NUM_DENSE} dense + {NUM_SPARSE} sparse"
    )
    return df


def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=val_frac + test_frac, random_state=seed)
    val, test   = train_test_split(temp, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    print(f"Split → Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ─── PyTorch Dataset ───────────────────────────────────────────────────────────

class CriteoStyleDataset(Dataset):
    """
    (dense_features, sparse_features, label) dataset.
    dense_features : float32 tensor of shape (NUM_DENSE,)
    sparse_features: int64  tensor of shape (NUM_SPARSE,)
    label          : float32 scalar
    """

    def __init__(self, df: pd.DataFrame):
        dense_cols  = [c for c in df.columns if c.startswith("dense_")]
        sparse_cols = SPARSE_KEYS

        self.dense  = torch.as_tensor(df[dense_cols].values,  dtype=torch.float32)
        self.sparse = torch.as_tensor(df[sparse_cols].values, dtype=torch.long)
        self.labels = torch.as_tensor(df["label"].values,     dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.dense[idx], self.sparse[idx], self.labels[idx]


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 4096,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    kw  = dict(num_workers=num_workers, pin_memory=pin)
    return (
        DataLoader(CriteoStyleDataset(train_df), batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(CriteoStyleDataset(val_df),   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(CriteoStyleDataset(test_df),  batch_size=batch_size, shuffle=False, **kw),
    )
