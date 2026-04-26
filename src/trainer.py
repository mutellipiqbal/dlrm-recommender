"""
Trainer — src/trainer.py
=========================
Training loop for DLRM with MLflow 3.x tracking.
Uses AdamW + OneCycleLR (industry standard for ranking models).
"""

from __future__ import annotations
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
import mlflow.pytorch


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for dense, sparse, labels in loader:
        dense  = dense.to(device,  non_blocking=True)
        sparse = sparse.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(dense, sparse)
        loss   = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()

    return {
        "loss":      total_loss / len(loader),
        "auc":       roc_auc_score(all_labels, probs),
        "prauc":     average_precision_score(all_labels, probs),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for dense, sparse, labels in loader:
        dense  = dense.to(device,  non_blocking=True)
        sparse = sparse.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(dense, sparse)
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()

    return {
        "loss":  total_loss / len(loader),
        "auc":   roc_auc_score(all_labels, probs),
        "prauc": average_precision_score(all_labels, probs),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    run_name: str = "dlrm",
) -> nn.Module:
    """Full training loop with MLflow 3.x tracking."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
    )
    # OneCycleLR: widely used for ranking models — fast warmup, cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["epochs"],
        pct_start=0.05,         # 5% warmup
        anneal_strategy="cos",
    )
    # Weighted BCE: handle class imbalance (positive rate ~15%)
    pos_weight = torch.tensor([cfg.get("pos_weight", 5.0)], device=device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.set_experiment(cfg["mlflow_experiment"])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, list)})
        mlflow.log_params({"top_mlp_dims":    str(cfg.get("top_mlp_dims")),
                           "bottom_mlp_dims": str(cfg.get("bottom_mlp_dims"))})
        best_auc = 0.0

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
            val_metrics   = evaluate(model, val_loader, loss_fn, device)
            elapsed       = time.time() - t0

            print(
                f"Epoch {epoch:>3}/{cfg['epochs']} | "
                f"train_loss={train_metrics['loss']:.4f}  train_auc={train_metrics['auc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}  val_auc={val_metrics['auc']:.4f}  "
                f"val_prauc={val_metrics['prauc']:.4f} | {elapsed:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss":  train_metrics["loss"],
                    "train_auc":   train_metrics["auc"],
                    "val_loss":    val_metrics["loss"],
                    "val_auc":     val_metrics["auc"],
                    "val_prauc":   val_metrics["prauc"],
                    "lr":          scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                torch.save(model.state_dict(), "best_dlrm.pt")
                mlflow.log_artifact("best_dlrm.pt")

        mlflow.log_metric("best_val_auc", best_auc)
        print(f"\nBest val AUC: {best_auc:.4f}")

    model.load_state_dict(torch.load("best_dlrm.pt", map_location=device))
    return model
