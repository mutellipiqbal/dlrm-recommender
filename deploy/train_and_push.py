"""
deploy/train_and_push.py  —  DLRM
===================================
Pushes the trained DLRM model weights + config to Hugging Face Hub.

Usage:
  1. pip install huggingface_hub
  2. huggingface-cli login
  3. python deploy/train_and_push.py
"""

import json
import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "YOUR_HF_USERNAME"
REPO_NAME   = "dlrm-reranker"
MODEL_PT    = "best_dlrm.pt"
META_JSON   = "dlrm_meta.json"

# Must match the config used during training
meta = {
    "num_dense_features": 13,
    "vocab_sizes": [100000, 50000, 500, 5000, 200, 5],
    "embedding_dim":    16,
    "bottom_mlp_dims":  [64, 16],
    "top_mlp_dims":     [256, 128, 64],
    "dropout":          0.1,
    "sparse_feature_names": [
        "user_id", "item_id", "category_id",
        "brand_id", "country_id", "device_type_id",
    ],
    "dense_feature_count": 13,
}

with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

repo_id = f"{HF_USERNAME}/{REPO_NAME}"
create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api = HfApi()

for path in [MODEL_PT, META_JSON]:
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        continue
    api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=repo_id)
    print(f"✓  {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

print(f"\nDone!  https://huggingface.co/{repo_id}")
