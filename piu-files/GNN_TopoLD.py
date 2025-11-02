import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
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
    GCN con feature topologiche e attributi degli archi.
    Usa data.x, data.topo, data.edge_attr.
    """
    def __init__(self, node_in_dim: int, topo_in_dim: int, edge_in_dim: int = 3,
                 hidden_dim: int = 64, topo_proj_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        self.topo = SimpleTopoLayer(topo_in_dim, proj_dim=topo_proj_dim, dropout=dropout)
        fused_in = node_in_dim + self.topo.out_dim

        # 🔹 MLP per gli attributi degli archi
        nn_edge = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fused_in * hidden_dim)
        )

        self.conv1 = NNConv(fused_in, hidden_dim, nn_edge, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn_edge, aggr='mean')
        self.dropout = nn.Dropout(dropout)
        self.out_lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        topo_z = self.topo(data)
        x = torch.cat([x, topo_z], dim=1)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)

        return torch.sigmoid(self.out_lin(x))
    
    
    def predict(self, data, threshold=0.5):
        self.eval()
        with torch.no_grad():
            probs = self.forward(data)
            preds = (probs > threshold).float()
        return preds, probs



