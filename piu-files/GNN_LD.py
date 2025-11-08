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


def train_model(model, optimizer, graphs, loss_fn=nn.BCELoss(), epochs=50, name="model"):
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

    for epoch in range(0, epochs):
        total_loss = 0.0
        model.train()

        for data in graphs:
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


class TopoDynamicUEstimator(nn.Module):
    """
    Implementa la dinamica di UdiK:
        x_{k+1} = M x_k + U_k
    Stima:
        U_hat = -ReLU(soft(-(x_{k+1} - Mx_k), tau))
    """
    def __init__(self, tau: float = 0.02):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=False)

    @staticmethod
    def soft(x, tau):
        return torch.sign(x) * torch.clamp(x.abs() - tau, min=0.0)

    def forward(self, data_t: Data, data_t1: Data):
        xk = data_t.edge_flow.view(-1, 1)
        xk1 = data_t1.edge_flow.view(-1, 1)
        M = data_t.M

        z = xk1 - (M @ xk)  # residuo dinamico
        U_hat = -F.relu(self.soft(-z, float(self.tau.item())))  # sparsità + segno fisico
        return U_hat, z

