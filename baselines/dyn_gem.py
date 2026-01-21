# baselines/dyn_gem.py

import numpy as np
import dgl
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


def run_dyn_gem(
    graphs,
    edge_labels,
    train_ratio=0.7,
    emb_dim=32,
    max_nodes=20000,
    seed=42
):
    """
    Sampled DynGEM baseline for large dynamic graphs.
    Edge-level anomaly detection with node-induced subgraphs.
    """

    rng = np.random.RandomState(seed)

    scores = []
    labels = []

    T = len(graphs)
    split = int(train_ratio * T)

    prev_emb = None
    prev_n = None

    for t in range(split, T):
        g_full = graphs[t]
        edge_label_full = edge_labels[t]

        n_full = g_full.num_nodes()

        # --------------------------------------------------
        # 1. Node sampling (build induced subgraph)
        # --------------------------------------------------
        if n_full > max_nodes:
            perm = rng.permutation(n_full)[:max_nodes]
            perm = np.sort(perm)
            g = dgl.node_subgraph(g_full, perm)

            # IMPORTANT: get edge labels aligned with subgraph
            # DGL keeps edge order; need to extract labels accordingly
            # Original edge ids corresponding to subgraph edges
            eids = g.edata[dgl.EID]
            edge_label = edge_label_full[eids]

        else:
            g = g_full
            edge_label = edge_label_full

        n = g.num_nodes()

        # --------------------------------------------------
        # 2. DynGEM-style adjacency embedding
        # --------------------------------------------------
        adj = g.adjacency_matrix().to_dense().numpy()

        pca = PCA(n_components=min(emb_dim, n - 1))
        emb = pca.fit_transform(adj)

        # --------------------------------------------------
        # 3. Temporal difference (node-aligned)
        # --------------------------------------------------
        if prev_emb is not None:
            n_common = min(prev_n, n)
            diff = np.linalg.norm(
                emb[:n_common] - prev_emb[:n_common],
                axis=1
            )
        else:
            n_common = n
            diff = np.zeros(n_common)

        # --------------------------------------------------
        # 4. Edge-level anomaly scoring (subgraph space)
        # --------------------------------------------------
        src, dst = g.edges()
        src = src.numpy()
        dst = dst.numpy()

        mask = (src < n_common) & (dst < n_common)
        src = src[mask]
        dst = dst[mask]

        edge_score = diff[src] + diff[dst]

        scores.extend(edge_score.tolist())
        labels.extend(edge_label[mask].numpy().tolist())

        prev_emb = emb
        prev_n = n

    return roc_auc_score(labels, scores)
