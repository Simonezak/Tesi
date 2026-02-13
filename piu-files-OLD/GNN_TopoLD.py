import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class SimpleTopoLayer(nn.Module):
    """
    Layer topologico leggero: proietta e normalizza feature topologiche pre-calcolate.
    Attende data.topo [N, topo_in_dim].
    """
    def __init__(self, topo_in_dim: int, proj_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(topo_in_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU()
        )
        self.out_dim = proj_dim

    def forward(self, data):
        topo = getattr(data, "topo", None)
        if topo is None:
            # Se il grafo non ha data.topo, restituisci zeri
            return torch.zeros((data.num_nodes, self.out_dim), device=data.x.device)
        return self.mlp(topo)


class GNNLeakDetectorTopo(nn.Module):
    """
    GCN per leak detection con fusione di feature topologiche pre-calcolate.
    Usa data.x (feature nodali) e data.topo (feature topologiche).
    """
    def __init__(self, node_in_dim: int, topo_in_dim: int, hidden_dim: int = 64,
                 topo_proj_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        # Topological feature encoder
        self.topo = SimpleTopoLayer(topo_in_dim, proj_dim=topo_proj_dim, dropout=dropout)

        # Dimensione combinata delle feature in ingresso
        fused_in = node_in_dim + self.topo.out_dim

        # Backbone GCN
        self.conv1 = GCNConv(fused_in, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 🔹 Calcola e concatena feature topologiche
        topo_z = self.topo(data)
        x = torch.cat([x, topo_z], dim=1)

        # 🔹 Message passing classico
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # 🔹 Output: probabilità di leak per nodo
        return torch.sigmoid(self.out_lin(x))

    def predict(self, data, threshold=0.5):
        self.eval()
        with torch.no_grad():
            probs = self.forward(data)
            preds = (probs > threshold).float()
        return preds, probs


