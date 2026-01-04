import os
import torch
import torch.nn as nn
import numpy as np
import networkx as nx

from GGNN_Regression import GGNNModel
from wntr_exp_Regression import WNTREnv, build_static_graph_from_wntr, build_attr_from_pressure_window
from main_dyn_topologyknown_01 import func_gen_B2_lu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        # ---- GGNN (node space)
        u_node = self.ggnn(attr_matrix, adj_matrix)
        u_node = u_node.view(-1)             # [N]

        # ---- node -> edge
        edge_feat = self.B1_T @ u_node       # [E]
        edge_feat = edge_feat.unsqueeze(-1)  # [E, 1]

        # ---- topological residual MLP
        edge_feat = self.topo(edge_feat)     # [E, 1]

        # ---- edge -> node
        u_node = self.B1 @ edge_feat.squeeze(-1)  # [N]
        return u_node


