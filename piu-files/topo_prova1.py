import os
import torch
import torch.nn as nn
import numpy as np
import networkx as nx

from GGNN_Regression import GGNNModel
from wntr_exp_Regression import (
    WNTREnv,
    build_static_graph_from_wntr,
    build_attr_from_pressure_window
)
from main_dyn_topologyknown_01 import func_gen_B2_lu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TOPOLOGICAL LAYER
# ============================================================

class TopologicalEdgeLayer(nn.Module):
    """
    X_out = X - alpha * L1 @ X
    """
    def __init__(self, L1, alpha=0.1):
        super().__init__()
        self.register_buffer("L1", torch.tensor(L1, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, X):
        return X - self.alpha * (self.L1 @ X)


# ============================================================
# GGNN + TOPOLOGY
# ============================================================

import torch
import torch.nn as nn

class GGNNWithTopology(nn.Module):
    def __init__(self, ggnn, topo_layer, B1_np):
        super().__init__()
        self.ggnn = ggnn
        self.topo = topo_layer

        B1 = torch.tensor(B1_np, dtype=torch.float32)      # [N, E]
        self.register_buffer("B1", B1)
        self.register_buffer("B1_T", B1.t())               # [E, N]

    def forward(self, attr_matrix, adj_matrix):
        u_node = self.ggnn(attr_matrix, adj_matrix)        # [1, N]
        u_node = u_node.view(-1)                           # [N]  (robusto)

        # Nodo -> Edge
        edge_feat = self.B1_T @ u_node                     # [E]
        edge_feat = edge_feat.unsqueeze(-1)                # [E, 1]

        # filtro topologico sugli edge
        edge_feat = self.topo(edge_feat)                   # [E, 1]

        # Edge -> Nodo
        u_node = self.B1 @ edge_feat.squeeze(-1)           # [N]

        return u_node




# ============================================================
# TRAINING
# ============================================================

def train():
    inp_path = "/home/zagaria/Tesi/Tesi/Networks-found/Jilin_copy_copy.inp"
    max_steps = 50
    window_size = 1
    epochs = 50
    lr = 1e-3
    area = 0.1

    env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    # --------- TOPOLOGY ----------
    G = nx.Graph()
    A_np = adj_matrix.cpu().numpy()[0]
    print(A_np.shape[0])
    print(A_np.shape[1])
    for i in range(A_np.shape[0]):
        for j in range(i + 1, A_np.shape[1]):
            if A_np[i, j] > 0:
                G.add_edge(i, j)

    B1, B2, _ = func_gen_B2_lu(G, max_cycle_length=8)
    L1 = B1.T @ B1 + B2 @ B2.T
    print(L1.shape)

    topo_layer = TopologicalEdgeLayer(L1).to(DEVICE)

    ggnn = GGNNModel(
        attr_size=window_size,
        hidden_size=210,
        propag_steps=6
    ).to(DEVICE)

    model = GGNNWithTopology(ggnn, topo_layer, B1).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    print("\n=== TRAINING TOPO-GGNN ===")

    for epoch in range(epochs):
        env.reset(num_leaks=2)
        sim = env.sim
        pressure_window = []

        for step in range(max_steps):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=area)

            sim.step_sim()
            results = sim.get_results()

            p = torch.tensor(
                results.node["pressure"]
                .iloc[-1][list(node2idx.keys())]
                .values,
                dtype=torch.float32
            ).to(DEVICE)

            pressure_window.append(p)

            if len(pressure_window) < window_size:
                continue
            if len(pressure_window) > window_size:
                pressure_window.pop(0)
            if step < env.leak_start_step:
                continue

            attr = build_attr_from_pressure_window(pressure_window).to(DEVICE)

            demand = results.node["demand"].iloc[-1][list(node2idx.keys())].values
            leak = results.node.get("leak_demand", None)
            leak = leak.iloc[-1][list(node2idx.keys())].values if leak is not None else 0
            target = torch.tensor(demand + leak, dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            out = model(attr, adj_matrix)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/topo_ggnn.pt")
    print("\n Modello salvato in saved_models/topo_ggnn.pt")


if __name__ == "__main__":
    train()
