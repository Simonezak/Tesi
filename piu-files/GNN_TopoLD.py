import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SimpleTopoLayer(nn.Module):
    """
    Layer topologico leggero: proietta e normalizza feature topologiche per nodo.
    Attende data.topo [N, topo_dim]. Se non presente, ignora (ritorna zeri).
    """
    def __init__(self, topo_in_dim: int, proj_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.has_topo = topo_in_dim > 0
        if self.has_topo:
            self.mlp = nn.Sequential(
                nn.Linear(topo_in_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU()
            )
            self.out_dim = proj_dim
        else:
            self.out_dim = 0

    def forward(self, data: Data):
        if not self.has_topo or getattr(data, "topo", None) is None:
            if self.out_dim == 0:
                return None
            return torch.zeros((data.num_nodes, self.out_dim), device=data.x.device)
        topo = data.topo
        return self.mlp(topo)


class GNNLeakDetectorTopo(nn.Module):
    """
    GCN per leak detection con fusione di un Topological Layer semplice.
    Output: prob. per nodo [N,1].
    """
    def __init__(self, node_in_dim: int, topo_in_dim: int = 0,
                 hidden_dim: int = 64, topo_proj_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        # Topo layer (stile TOGL: proietta e concatena)
        self.topo = SimpleTopoLayer(topo_in_dim, proj_dim=topo_proj_dim, dropout=dropout)
        fused_in = node_in_dim + (self.topo.out_dim if self.topo.out_dim else 0)

        # GCN backbone
        self.conv1 = GCNConv(fused_in, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_lin = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # 1) Topo features proiettate + concatenazione
        topo_z = self.topo(data)
        if topo_z is not None:
            x = torch.cat([x, topo_z], dim=1)

        # 2) GCN
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # 3) Output per-nodo
        return torch.sigmoid(self.out_lin(x))

    def predict(self, data: Data, threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            p = self.forward(data)
            return (p > threshold).float(), p

