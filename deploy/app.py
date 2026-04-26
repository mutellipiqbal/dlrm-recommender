"""
deploy/app.py  —  DLRM Re-Ranker  (Hugging Face Spaces / Gradio)
=================================================================
Demonstrates re-ranking a set of candidates with DLRM scores.
In production these candidates come from the Two-Tower FAISS retrieval step.

Deploy steps:
  1. Create HF Space → SDK: Gradio → Hardware: CPU Basic (free)
  2. Clone Space repo, copy this file as app.py + the src/ folder
  3. Set HF_REPO below
  4. git push → Space auto-builds
"""

from __future__ import annotations
import json
import sys
import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download

HF_REPO = "YOUR_HF_USERNAME/dlrm-reranker"   # ← edit this

sys.path.insert(0, ".")
from src.model   import DLRM
from src.dataset import NUM_DENSE, NUM_SPARSE, SPARSE_KEYS, VOCAB_SIZES

# ── Load at startup ───────────────────────────────────────────────────────────
print("Loading DLRM from Hub...")
meta     = json.loads(open(hf_hub_download(HF_REPO, "dlrm_meta.json")).read())
model_pt = hf_hub_download(HF_REPO, "best_dlrm.pt")

model = DLRM(
    num_dense_features = meta["num_dense_features"],
    vocab_sizes        = meta["vocab_sizes"],
    embedding_dim      = meta["embedding_dim"],
    bottom_mlp_dims    = meta["bottom_mlp_dims"],
    top_mlp_dims       = meta["top_mlp_dims"],
    dropout            = meta["dropout"],
)
model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
model.eval()
print("DLRM ready.")


# ── Inference ─────────────────────────────────────────────────────────────────
def rerank(n_candidates: int, seed: int) -> str:
    """
    Simulate receiving candidates from a Two-Tower retrieval stage and
    re-ranking them with DLRM. Returns a markdown ranking table.
    """
    rng = np.random.default_rng(seed)
    dense  = torch.tensor(rng.standard_normal((n_candidates, NUM_DENSE)), dtype=torch.float32)
    sparse = torch.tensor(
        np.column_stack([
            rng.integers(1, v + 1, size=n_candidates)
            for v in VOCAB_SIZES.values()
        ]),
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(dense, sparse)
        scores = torch.sigmoid(logits).numpy()

    ranked_idx = scores.argsort()[::-1]
    lines = [
        "| Rank | Candidate | DLRM Score |",
        "|------|-----------|------------|",
    ]
    for rank, idx in enumerate(ranked_idx, 1):
        lines.append(f"| {rank} | Item {idx + 1:>4} | {scores[idx]:.4f} |")

    return "\n".join(lines)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DLRM Re-Ranker") as demo:
    gr.Markdown(
        "# 🎯 DLRM Re-Ranker\n"
        "Simulates receiving candidates from a Two-Tower retrieval stage "
        "and re-ranking them using Deep Learning Recommendation Model (DLRM).\n\n"
        "In production the candidates come from a FAISS ANN search over item embeddings."
    )
    with gr.Row():
        n_cands = gr.Slider(label="Number of candidates", minimum=10, maximum=100, step=10, value=20)
        seed_in = gr.Number(label="Random seed", value=42, precision=0)
    btn    = gr.Button("Re-rank candidates", variant="primary")
    output = gr.Markdown()
    btn.click(fn=rerank, inputs=[n_cands, seed_in], outputs=output)

    gr.Markdown(
        "### Architecture\n"
        "```\n"
        "13 dense features  → Bottom MLP → 16-dim\n"
        " 6 sparse features → Embedding tables → 16-dim each\n"
        "Feature interaction (dot product pairs) → Top MLP → P(click)\n"
        "```"
    )

if __name__ == "__main__":
    demo.launch()
