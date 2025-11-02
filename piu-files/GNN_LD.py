import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


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


def train_model(model, optimizer, graphs, loss_fn=None, epochs=50, name="model"):
    """
    Allena un modello GNN supervisionato su una lista di grafi PyG.

    Args:
        model: istanza della rete neurale PyTorch
        optimizer: ottimizzatore (es. Adam)
        graphs: lista di oggetti torch_geometric.data.Data
        labels: lista di tensori [num_nodes, 1]
        loss_fn: funzione di loss (default: BCELoss)
        epochs: numero di epoche
        name: stringa descrittiva (per logging)
    """
    
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()

        for data in graphs:
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()




