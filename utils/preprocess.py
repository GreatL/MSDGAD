import pandas as pd
import dgl
import torch

def load_bitcoin_alpha(path, time_window=14 * 24 * 3600):
    df = pd.read_csv(
        path,
        names=["src", "dst", "rating", "timestamp"]
    )

    df = df.sort_values("timestamp")
    t_min = df["timestamp"].min()
    df["snapshot"] = ((df["timestamp"] - t_min) // time_window).astype(int)

    graphs = []
    edge_labels = []

    for _, sub in df.groupby("snapshot"):
        src = torch.tensor(sub["src"].values)
        dst = torch.tensor(sub["dst"].values)

        # ✅ 不做 to_simple，保留多重边
        g = dgl.graph((src, dst))

        # ✅ self-loop（label=0）
        num_nodes = g.num_nodes()
        self_src = torch.arange(num_nodes)
        self_dst = torch.arange(num_nodes)
        g = dgl.add_edges(g, self_src, self_dst)

        # ✅ node features
        deg_in = g.in_degrees().float()
        deg_out = g.out_degrees().float()
        g.ndata["feat"] = torch.stack([deg_in, deg_out], dim=1)

        # ✅ edge labels：一一对应
        rating = torch.tensor(sub["rating"].values)
        edge_label = (rating < 0).float()

        # self-loop 的 label = 0
        self_loop_label = torch.zeros(num_nodes)
        edge_label = torch.cat([edge_label, self_loop_label], dim=0)

        graphs.append(g)
        edge_labels.append(edge_label)

    return graphs, edge_labels
