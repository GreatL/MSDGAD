from torch.utils.data import Dataset

class DynamicGraphDataset(Dataset):
    def __init__(self, graphs, edge_labels, window_size=3):
        self.graphs = graphs
        self.edge_labels = edge_labels
        self.window_size = window_size

    def __len__(self):
        # 用 [t-2, t-1, t] 预测 t+1
        return len(self.graphs) - self.window_size

    def __getitem__(self, idx):
        g_seq = self.graphs[idx : idx + self.window_size]
        g_next = self.graphs[idx + self.window_size]
        edge_label = self.edge_labels[idx + self.window_size]
        return g_seq, g_next, edge_label
