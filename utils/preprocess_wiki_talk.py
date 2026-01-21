import pandas as pd
import torch
import dgl
from collections import defaultdict


def load_wiki_talk(
    path,
    time_window=14 * 24 * 3600,
    freq_threshold=1
):
    """
    Wikipedia Talk temporal network
    Anomalous edge = rare user-user interaction
    """

    df = pd.read_csv(
        path,
        sep=" ",
        names=["src", "dst", "timestamp"]
    )

    df = df.sort_values("timestamp")
    t_min = df["timestamp"].min()
    df["snapshot"] = ((df["timestamp"] - t_min) // time_window).astype(int)

    graphs = []
    edge_labels = []

    edge_freq = defaultdict(int)

    for _, sub in df.groupby("snapshot"):
        src = torch.tensor(sub["src"].values)
        dst = torch.tensor(sub["dst"].values)

        g = dgl.graph((src, dst))

        # self-loop
        num_nodes = g.num_nodes()
        g = dgl.add_edges(g, torch.arange(num_nodes), torch.arange(num_nodes))

        # node features: degree
        deg_in = g.in_degrees().float()
        deg_out = g.out_degrees().float()
        g.ndata["feat"] = torch.stack([deg_in, deg_out], dim=1)

        labels = []
        for s, d in zip(src.tolist(), dst.tolist()):
            key = (s, d)
            labels.append(1.0 if edge_freq[key] <= freq_threshold else 0.0)
            edge_freq[key] += 1

        # self-loop label = 0
        labels += [0.0] * num_nodes

        graphs.append(g)
        edge_labels.append(torch.tensor(labels))

    return graphs, edge_labels
