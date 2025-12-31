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


# ============================================================
# TRAINING
# ============================================================

def train():
    inp_path = "/home/zagaria/Tesi/Tesi/Networks-found/Jilin_copy_copy.inp"

    # ====== IPERPARAMETRI ======
    MAX_STEPS = 50
    WINDOW_SIZE = 4
    EPOCHS = 400
    LR = 1e-3
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6
    TOPO_MLP_HIDDEN = 16

    # ====== ENV & GRAPH ======
    env = WNTREnv(inp_path, max_steps=MAX_STEPS)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    # ====== BUILD TOPOLOGY ======
    G = nx.Graph()
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        u = pipe.start_node_name
        v = pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    B1, B2, _ = func_gen_B2_lu(G, max_cycle_length=8)
    L1 = B1.T @ B1 + B2 @ B2.T

    # ====== MODEL ======
    ggnn = GGNNModel(
        attr_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    ).to(DEVICE)

    topo_layer = TopoResidual(L1, hidden=TOPO_MLP_HIDDEN).to(DEVICE)
    model = GGNNWithTopoMLP(ggnn, topo_layer, B1).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING TOPO-MLP-GGNN ===")

    # ====== TRAIN LOOP ======
    for epoch in range(EPOCHS):
        env.reset(num_leaks=2)
        sim = env.sim
        pressure_window = []

        for step in range(MAX_STEPS):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=LEAK_AREA)

            sim.step_sim()
            results = sim.get_results()

            p = torch.tensor(
                results.node["pressure"]
                .iloc[-1][list(node2idx.keys())]
                .values,
                dtype=torch.float32
            ).to(DEVICE)

            pressure_window.append(p)

            if len(pressure_window) < WINDOW_SIZE:
                continue
            if len(pressure_window) > WINDOW_SIZE:
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

        if epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")

    # ====== SAVE ======
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/topo_mlp_ggnn.pt")
    print("\n✅ Modello Topo-MLP salvato in saved_models/topo_mlp_ggnn.pt")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()
