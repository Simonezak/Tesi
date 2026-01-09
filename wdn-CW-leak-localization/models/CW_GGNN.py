
import torch
import torch.nn as nn

# ============================================================
# TOPOLOGICAL RESIDUAL MLP LAYER
# ============================================================

class TopoResidual(nn.Module):
    """
    x_out = x + MLP(L1 x)
    """
    def __init__(self, L1, hidden=16):
        super().__init__()
        self.register_buffer("L1", torch.tensor(L1, dtype=torch.float32))

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, X):
        # X: [E, 1]
        topo_feat = self.L1 @ X          # [E, 1]
        return X + self.mlp(topo_feat)   # residual correction


# ============================================================
# GGNN + TOPO-MLP
# ============================================================

class GGNNWithTopoMLP(nn.Module):
    def __init__(self, ggnn, topo_layer, B1_np):
        super().__init__()
        self.ggnn = ggnn
        self.topo = topo_layer

        B1 = torch.tensor(B1_np, dtype=torch.float32)
        self.register_buffer("B1", B1)       # [N, E]
        self.register_buffer("B1_T", B1.t()) # [E, N]

    def forward(self, attr_matrix, adj_matrix):
        u_node = self.ggnn(attr_matrix, adj_matrix)
        edge_feat = self.B1_T @ u_node
        edge_feat = edge_feat.unsqueeze(-1)
        edge_feat = self.topo(edge_feat)
        u_node = self.B1 @ edge_feat.squeeze(-1)

        return u_node


