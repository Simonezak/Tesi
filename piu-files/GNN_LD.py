import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GNNLeakDetector(nn.Module):
    """
    Simple GNN for Leak Detection (node-level classification).
    Each node receives a probability of leak (0=no leak, 1=leak).
    """

    def __init__(self, node_in_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        # Layers
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        """
        Args:
            data (torch_geometric.data.Data): graph with
                - x : node features [num_nodes, node_in_dim]
                - edge_index : [2, num_edges]
        Returns:
            torch.Tensor : leak probabilities per node [num_nodes, 1]
        """
        x, edge_index = data.x, data.edge_index

        # 1st GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # 2nd GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Output (per-node leak probability)
        out = torch.sigmoid(self.linear_out(x))
        return out

    def predict(self, data: Data, threshold: float = 0.5):
        """
        Returns binary predictions for leak presence.
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(data)
            return (probs > threshold).float(), probs
