# baselines/lof.py
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

def run_lof(graphs, edge_labels, train_ratio=0.7):
    scores = []
    labels = []

    T = len(graphs)
    split = int(train_ratio * T)

    rng = np.random.RandomState(42)

    for t in range(split, T):
        g = graphs[t]
        src, dst = g.edges()
        deg = g.in_degrees().numpy()

        X = np.stack([
            deg[src.numpy()],
            deg[dst.numpy()]
        ], axis=1).astype(float)

        # ✅ small noise to break ties
        X += 1e-3 * rng.randn(*X.shape)

        lof = LocalOutlierFactor(
            n_neighbors=35,
            novelty=False
        )

        lof.fit(X)
        score = -lof.negative_outlier_factor_

        scores.extend(score)
        labels.extend(edge_labels[t].numpy())

    return roc_auc_score(labels, scores)
