import dgl.nn as dglnn
import torch.nn as nn

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.gcn = dglnn.GraphConv(in_dim, hid_dim)

    def forward(self, g, x):
        return self.gcn(g, x)
