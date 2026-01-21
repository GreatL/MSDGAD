import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import dgl
import numpy as np
import torch

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    count = 0

    for g_seq, g_next, edge_label in loader:
        """
        g_seq: list of graphs [g_{t-2}, g_{t-1}, g_t]
        g_next: graph g_{t+1}
        edge_label: edge labels for g_{t+1}
        """

        # =====================================================
        # 1. Node alignment across g_seq and g_next
        # =====================================================
        num_nodes_list = [g.num_nodes() for g in g_seq]
        num_nodes_list.append(g_next.num_nodes())
        n_common = min(num_nodes_list)

        # =====================================================
        # 2. Filter edges in g_next (only keep aligned nodes)
        # =====================================================
        src, dst = g_next.edges()
        mask = (src < n_common) & (dst < n_common)

        if mask.sum() == 0:
            continue

        src = src[mask]
        dst = dst[mask]
        edge_label = edge_label[mask]

        # =====================================================
        # 3. Build aligned g_next'
        # =====================================================
        g_next_aligned = dgl.graph((src, dst), num_nodes=n_common)

        # =====================================================
        # 4. Forward + loss
        # =====================================================
        pred = model(g_seq, g_next_aligned)

        loss = F.binary_cross_entropy_with_logits(
            pred, edge_label
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    scores = []
    labels = []

    for g_seq, g_next, edge_label in loader:
        # =====================================================
        # 1. Node alignment
        # =====================================================
        num_nodes_list = [g.num_nodes() for g in g_seq]
        num_nodes_list.append(g_next.num_nodes())
        n_common = min(num_nodes_list)

        # =====================================================
        # 2. Filter edges
        # =====================================================
        src, dst = g_next.edges()
        mask = (src < n_common) & (dst < n_common)

        if mask.sum() == 0:
            continue

        src = src[mask]
        dst = dst[mask]
        edge_label = edge_label[mask]

        # =====================================================
        # 3. Build aligned g_next'
        # =====================================================
        g_next_aligned = dgl.graph((src, dst), num_nodes=n_common)

        # =====================================================
        # 4. Forward
        # =====================================================
        pred = model(g_seq, g_next_aligned)

        scores.extend(pred.cpu().numpy())
        labels.extend(edge_label.cpu().numpy())

    if len(set(labels)) < 2:
        return 0.5

    return roc_auc_score(labels, scores)

@torch.no_grad()
def evaluate_and_collect(model, loader, save_path,
                          max_samples=20000):
    """
    Evaluate model on test set and collect anomaly scores and labels
    for visualization (Fig.4).

    Parameters
    ----------
    model : torch.nn.Module
    loader : DataLoader
        Test dataloader
    save_path : str
        Path to save npz file
    max_samples : int
        Maximum number of edges to save
    """

    model.eval()

    all_scores = []
    all_labels = []

    for g_seq, g_next, edge_label in loader:

        # ------------------------------
        # Node alignment (same as evaluate)
        # ------------------------------
        num_nodes_list = [g.num_nodes() for g in g_seq]
        num_nodes_list.append(g_next.num_nodes())
        n_common = min(num_nodes_list)

        src, dst = g_next.edges()
        mask = (src < n_common) & (dst < n_common)

        if mask.sum() == 0:
            continue

        src = src[mask]
        dst = dst[mask]
        labels = edge_label[mask].cpu().numpy()

        g_next_aligned = dgl.graph((src, dst), num_nodes=n_common)

        # ------------------------------
        # Forward
        # ------------------------------
        logits = model(g_seq, g_next_aligned)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels)

        # ------------------------------
        # Early stop if too many samples
        # ------------------------------
        if sum(len(s) for s in all_scores) >= max_samples:
            break

    # ------------------------------
    # Concatenate & subsample
    # ------------------------------
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    if len(all_scores) > max_samples:
        idx = np.random.choice(len(all_scores), max_samples, replace=False)
        all_scores = all_scores[idx]
        all_labels = all_labels[idx]

    # ------------------------------
    # Save
    # ------------------------------
    np.savez(
        save_path,
        scores=all_scores,
        labels=all_labels
    )

    print(f"[Info] Saved {len(all_scores)} samples to {save_path}")
