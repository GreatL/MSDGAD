import torch
import torch.nn as nn

class MultiScaleTemporal(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        # short-term
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True)

        # long-term (simple attention)
        self.attn = nn.Linear(hid_dim, 1)

        self.fusion = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, h_seq):
        """
        h_seq: [N, T, D]  (T=1 in current setting, extensible)
        """
        # short-term
        h_short, _ = self.gru(h_seq)
        h_short = h_short[:, -1, :]

        # long-term
        w = torch.softmax(self.attn(h_seq), dim=1)
        h_long = (w * h_seq).sum(dim=1)

        h = torch.cat([h_short, h_long], dim=1)
        return self.fusion(h)
