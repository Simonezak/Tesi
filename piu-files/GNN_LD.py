import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool


class GNNLeakDetector(nn.Module):
    """
    GNN per rilevazione perdite (solo nodi), con edge_attr.
    """
    def __init__(self, node_in_dim: int = 4, edge_in_dim: int = 3,
                 hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        # 🔹 MLP per proiettare edge_attr -> pesi dinamici per NNConv
        nn_edge = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_in_dim * hidden_dim)
        )

        self.conv1 = NNConv(node_in_dim, hidden_dim, nn_edge, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn_edge, aggr='mean')
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)

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




