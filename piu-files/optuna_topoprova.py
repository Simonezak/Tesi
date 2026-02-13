import optuna
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
from test_model import ranking_score_lexicographic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TOPOLOGICAL LAYER
# ============================================================

class TopologicalEdgeLayer(nn.Module):
    def __init__(self, L1, alpha):
        super().__init__()
        self.register_buffer("L1", torch.tensor(L1, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, X):
        return X - self.alpha * (self.L1 @ X)


class GGNNWithTopology(nn.Module):
    def __init__(self, ggnn, topo_layer, B1_np):
        super().__init__()
        self.ggnn = ggnn
        self.topo = topo_layer

        B1 = torch.tensor(B1_np, dtype=torch.float32)
        self.register_buffer("B1", B1)
        self.register_buffer("B1_T", B1.t())

    def forward(self, attr_matrix, adj_matrix):
        u_node = self.ggnn(attr_matrix, adj_matrix).view(-1)
        edge_feat = self.B1_T @ u_node
        edge_feat = edge_feat.unsqueeze(-1)
        edge_feat = self.topo(edge_feat)
        u_node = self.B1 @ edge_feat.squeeze(-1)
        return u_node


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # =========================
    # IPERPARAMETRI
    # =========================
    hidden_size  = trial.suggest_int("hidden_size", 64, 256)
    propag_steps = trial.suggest_int("propag_steps", 2, 7)
    #window_size  = trial.suggest_int("window_size", 2, 6)
    window_size  = 4
    lr           = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    alpha        = trial.suggest_float("alpha", 0.01, 0.5)
    #epochs       = trial.suggest_int("epochs", 80, 200)
    epochs       = 30

    inp_path  = "/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    max_steps = 50
    area      = 0.1

    # =========================
    # BUILD ENV + GRAPH
    # =========================
    env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    # =========================
    # TOPOLOGIA
    # =========================
    G = nx.Graph()
    A = adj_matrix.cpu().numpy()[0]
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j] > 0:
                G.add_edge(i, j)

    B1, B2, _ = func_gen_B2_lu(G, max_cycle_length=8)
    L1 = B1.T @ B1 + B2 @ B2.T

    topo_layer = TopologicalEdgeLayer(L1, alpha).to(DEVICE)

    ggnn = GGNNModel(
        attr_size=window_size,
        hidden_size=hidden_size,
        propag_steps=propag_steps
    ).to(DEVICE)

    model = GGNNWithTopology(ggnn, topo_layer, B1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # =====================================================
    # TRAINING — 1 SAMPLE ALLA VOLTA
    # =====================================================
    model.train()

    for _ in range(epochs):

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
                results.node["pressure"].iloc[-1][list(node2idx.keys())].values,
                dtype=torch.float32
            ).to(DEVICE)

            pressure_window.append(p)
            if len(pressure_window) > window_size:
                pressure_window.pop(0)
            if len(pressure_window) < window_size or step < env.leak_start_step:
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

    # =====================================================
    # TEST PHASE (COME test_model.py)
    # =====================================================
    model.eval()

    env.reset(num_leaks=2)
    sim = env.sim
    pressure_window = []
    anomaly_ts = []

    for step in range(max_steps):

        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(ln, leak_area=area)

        sim.step_sim()
        results = sim.get_results()

        p = torch.tensor(
            results.node["pressure"].iloc[-1][list(node2idx.keys())].values,
            dtype=torch.float32
        )

        pressure_window.append(p)
        if len(pressure_window) > window_size:
            pressure_window.pop(0)
        if len(pressure_window) < window_size or step < env.leak_start_step:
            continue

        attr = build_attr_from_pressure_window(pressure_window).to(DEVICE)

        with torch.no_grad():
            u_pred = model(attr, adj_matrix).cpu().numpy()

        anomaly_ts.append(u_pred)

    A = np.array(anomaly_ts)       # [T, N]
    score_per_node = A.sum(axis=0) # [N]

    score = ranking_score_lexicographic(
        score_per_node,
        idx2node,
        env.leak_node_names
    )

    return float(score)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="TopoGGNN_score_based",
        storage="sqlite:///optuna_topo_score.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=30)


    print("\n=== OPTUNA FINISHED ===")
    print("Best loss:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
