# DLRM Re-Ranking Model 🎯

> **Stage in the recommendation funnel:** Re-Ranking
> Takes ~100 candidates from retrieval and scores them precisely to return the final top-10.

Standalone PyTorch port of the [Databricks DLRM reference](https://docs.databricks.com/aws/en/machine-learning/train-recommender-models).
**No Spark. No Databricks. Runs free on Google Colab T4 or Kaggle P100.**

---

## Architecture

```
Dense  (13 features) ─► Bottom MLP ─────────────────────────────────┐
  user_age, price,        [13 → 64 → 16]                            │
  click_rate, …                                                      ├─► Feature Interaction ─► Top MLP ─► P(click)
                                                                     │     dot-product pairs    [37→256→128→64→1]
Sparse (6 features) ─► Embedding Tables ─► [e1, e2, e3, e4, e5, e6]┘
  user_id, item_id,       (each 16-dim)
  category_id, brand_id,
  country_id, device_type
```

- **Feature interaction:** dot-product between all pairs of (dense_proj + 6 sparse embs) = 21 pairs
- **Training:** BCEWithLogitsLoss with `pos_weight` for class imbalance, AdamW + OneCycleLR
- **Metrics:** AUC-ROC + PR-AUC (both important for imbalanced ranking)

---

## Library Versions (April 2026)

| Library | Version | Why |
|---|---|---|
| `torch` | 2.11.0 | `torch.compile` + latest CUDA kernels |
| `mlflow` | 3.11.1 | Experiment tracking, model registry |
| `scikit-learn` | 1.8.0 | AUC-ROC, PR-AUC metrics |
| `pandas` | 3.0.2 | Data handling |
| `numpy` | 2.2.5 | Synthetic data generation |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/dlrm-recommender.git
cd dlrm-recommender
pip install -r requirements.txt
jupyter notebook dlrm_recommender.ipynb
```

Or open directly in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/dlrm-recommender/blob/main/dlrm_recommender.ipynb)

---

## Project Structure

```
dlrm-recommender/
├── dlrm_recommender.ipynb   ← Main notebook (start here)
├── requirements.txt
├── src/
│   ├── model.py     ← DLRM, MLP, SparseEmbeddings, FeatureInteraction
│   ├── dataset.py   ← Criteo-style synthetic data, DataLoader
│   └── trainer.py   ← Training loop with MLflow 3.x
└── README.md
```

---

## What was changed from the Databricks original

| Databricks | This repo |
|---|---|
| `TorchDistributor` (requires PySpark cluster) | Standard `torch.compile` + single GPU |
| `StreamingDataset` (requires S3/DBFS) | `torch.utils.data.DataLoader` |
| `TorchRec` sharded embedding tables | `nn.Embedding` per field |
| `dbutils` / `spark.sql` | Removed |
| Databricks-hosted MLflow | Open-source `mlflow==3.11.1` |
| Synthetic Delta table | NumPy-generated (same schema: 13 dense + 6 sparse) |

---

## Free GPU Platforms

| Platform | GPU | Free Quota | Notes |
|---|---|---|---|
| Google Colab | T4 (16 GB) | ~12 hrs/session | Fastest to start |
| Kaggle Notebooks | P100 (16 GB) | 30 hrs/week | Best for reproducibility |
| Paperspace Gradient | M4000 (8 GB) | Free tier | Persistent storage |
| Lightning.ai | T4 | 22 hrs/month | Good MLflow UI |

---

## End-to-End Pipeline

```
All Items (50,000)
       │
       ▼
[Two-Tower Retrieval] ─── FAISS ANN ───► top-100 candidates  (fast, ~ms)
       │
       ▼
[DLRM Re-Ranker] ─── score each ───► top-10 final items  (precise, ~10ms)
```

See companion project: **`two-tower-recommender`** for the retrieval stage.
