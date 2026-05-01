"""Retrieval and clustering evaluation for the trained model."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from src.dataset import AbjadAudioDataset, collate_fn
from src.model import WavLMSupConModel


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(matrix, axis=0)) / np.sum(matrix))


def extract_embeddings(model, dataloader, device: str = "cuda"):
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].cpu().numpy()
            embs = model(input_values).cpu().numpy()
            all_embs.append(embs)
            all_labels.extend(labels)
    return np.vstack(all_embs), np.array(all_labels)


def retrieval_accuracy(embs, labels, top_ks=(1, 5)):
    """Compute top-k retrieval accuracy.

    A query is considered correct at rank k if any of the top-k nearest
    neighbours (excluding itself) shares the query's label.
    """
    sims = cosine_similarity(embs)
    np.fill_diagonal(sims, -np.inf)
    rankings = np.argsort(-sims, axis=1)

    results = {}
    for k in top_ks:
        topk = rankings[:, :k]
        correct = np.array([labels[i] in labels[topk[i]] for i in range(len(labels))])
        results[f"top_{k}"] = float(correct.mean())
    return results


def same_diff_similarity(embs, labels):
    """Mean cosine similarity for same-class and different-class pairs."""
    sims = cosine_similarity(embs)
    n = len(labels)
    same_vals, diff_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same_vals.append(sims[i, j])
            else:
                diff_vals.append(sims[i, j])
    same_mean = float(np.mean(same_vals))
    diff_mean = float(np.mean(diff_vals))
    return {
        "same_word_sim": same_mean,
        "diff_word_sim": diff_mean,
        "gap": same_mean - diff_mean,
    }


def agglomerative_eval(embs, labels):
    n_clusters = len(set(labels))
    pred = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(embs)
    return {
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "purity": purity_score(labels, pred),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    test_df,
    sample_rate: int = 16000,
    max_seconds: float = 2.5,
    samples_per_class: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
):
    eval_df = (
        test_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42))
        .reset_index(drop=True)
    )
    max_len = int(sample_rate * max_seconds)
    dataset = AbjadAudioDataset(eval_df, sr=sample_rate, max_len=max_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = WavLMSupConModel().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    embs, labels = extract_embeddings(model, loader, device=device)

    return {
        "n_units": int(len(labels)),
        "n_classes": int(len(set(labels))),
        "retrieval": retrieval_accuracy(embs, labels, top_ks=(1, 5)),
        "similarity": same_diff_similarity(embs, labels),
        "clustering": agglomerative_eval(embs, labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test_csv", required=True, help="CSV with test split")
    args = parser.parse_args()

    import pandas as pd

    test_df = pd.read_csv(args.test_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_checkpoint(args.checkpoint, test_df, device=device)

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
