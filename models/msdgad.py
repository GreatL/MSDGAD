import torch
import torch.nn as nn
import dgl

from models.gnn import GCNEncoder
from models.temporal import MultiScaleTemporal


class MSDGAD(nn.Module):
    def __init__(self, in_dim, hid_dim, use_temporal=True):
        super().__init__()
        self.use_temporal = use_temporal

        self.encoder = GCNEncoder(in_dim, hid_dim)

        if use_temporal:
            self.temporal = MultiScaleTemporal(hid_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, g_seq, g_next):
        """
        g_seq: list of graphs [g_{t-2}, g_{t-1}, g_t]
        g_next: already node-aligned graph (n_common nodes)
        """

        # =====================================================
        # 1. Determine common node set size
        # =====================================================
        num_nodes_list = [g.num_nodes() for g in g_seq]
        n_common = min(num_nodes_list)

        # =====================================================
        # 2. Encode each snapshot on node-aligned subgraph
        # =====================================================
        h_list = []
        for g in g_seq:
            # ---- key fix: build induced subgraph ----
            g_sub = dgl.node_subgraph(g, torch.arange(n_common))

            x = g_sub.ndata["feat"]
            h = self.encoder(g_sub, x)
            h_list.append(h)

        # [N_common, T, D]
        h_seq = torch.stack(h_list, dim=1)

        # =====================================================
        # 3. Temporal modeling
        # =====================================================
        if self.use_temporal:
            h = self.temporal(h_seq)
        else:
            h = h_seq[:, -1, :]

        # =====================================================
        # 4. Edge-level scoring
        # =====================================================
        src, dst = g_next.edges()
        h_src = h[src]
        h_dst = h[dst]

        return self.edge_mlp(
            torch.cat([h_src, h_dst], dim=1)
        ).squeeze()
