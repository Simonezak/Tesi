import torch
import numpy as np
import networkx as nx

from wntr_exp_Regression import (
    WNTREnv,
    build_static_graph_from_wntr,
    build_attr_from_pressure_window
)

from GGNN_Regression import GGNNModel
from main_dyn_topologyknown_01 import func_gen_B2_lu
from optuna_GGNN import ranking_score_lexicographic


DEVICE = "cpu"


# ============================================================
# TOPOLOGICAL RESIDUAL MLP LAYER
# ============================================================

class TopoResidual(torch.nn.Module):
    """
    x_out = x + MLP(L1 x)
    """
    def __init__(self, L1, hidden=16):
        super().__init__()
        self.register_buffer("L1", torch.tensor(L1, dtype=torch.float32))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, X):
        topo_feat = self.L1 @ X          # [E,1]
        return X + self.mlp(topo_feat)   # residual


# ============================================================
# GGNN + TOPO-MLP
# ============================================================

class GGNNWithTopoMLP(torch.nn.Module):
    def __init__(self, ggnn, topo_layer, B1_np):
        super().__init__()
        self.ggnn = ggnn
        self.topo = topo_layer

        B1 = torch.tensor(B1_np, dtype=torch.float32)
        self.register_buffer("B1", B1)
        self.register_buffer("B1_T", B1.t())

    def forward(self, attr_matrix, adj_matrix):
        u_node = self.ggnn(attr_matrix, adj_matrix)
        u_node = u_node.view(-1)              # [N]

        edge_feat = self.B1_T @ u_node        # [E]
        edge_feat = edge_feat.unsqueeze(-1)

        edge_feat = self.topo(edge_feat)      # [E,1]

        u_node = self.B1 @ edge_feat.squeeze(-1)
        return u_node


# ============================================================
# LOAD MODEL
# ============================================================

def load_topo_mlp_model(ckpt_path, inp_path):
    """
    Carica SOLO il modello Topo-MLP-GGNN.
    Iperparametri coerenti col training.
    """

    # ===== IPERPARAMETRI (DEVONO COINCIDERE) =====
    WINDOW_SIZE = 4
    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6
    TOPO_MLP_HIDDEN = 16

    # ---- GGNN ----
    ggnn = GGNNModel(
        attr_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    )

    # ---- Build topology ----
    env_tmp = WNTREnv(inp_path, max_steps=10)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env_tmp.wn)

    G = nx.Graph()
    for pipe_name in env_tmp.wn.pipe_name_list:
        pipe = env_tmp.wn.get_link(pipe_name)
        u = pipe.start_node_name
        v = pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    B1, B2, _ = func_gen_B2_lu(G, max_cycle_length=8)
    L1 = B1.T @ B1 + B2 @ B2.T

    topo_layer = TopoResidual(L1, hidden=TOPO_MLP_HIDDEN)

    model = GGNNWithTopoMLP(ggnn, topo_layer, B1)

    # ---- LOAD WEIGHTS ----
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    print("[OK] Topo-MLP-GGNN caricato correttamente")

    return model


# ============================================================
# SINGLE TEST EPISODE
# ============================================================

def run_single_test_episode(
    inp_path,
    model,
    max_steps,
    window_size,
    leak_area
):
    env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    env.reset(num_leaks=2)
    sim = env.sim

    # ------------------------
    # Run WNTR simulation
    # ------------------------
    for step in range(max_steps):
        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(
                    ln,
                    leak_area=leak_area,
                    leak_discharge_coefficient=0.75
                )
        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]
    cols = list(node2idx.keys())

    # ------------------------
    # Leak localization ONLY
    # ------------------------
    pressure_window = []
    anomaly_time_series = []

    for t in range(env.leak_start_step, len(df_pressure)):
        p = torch.tensor(
            df_pressure.iloc[t][cols].to_numpy(dtype=np.float32)
        )
        pressure_window.append(p)

        if len(pressure_window) < window_size:
            continue
        if len(pressure_window) > window_size:
            pressure_window.pop(0)

        attr = build_attr_from_pressure_window(pressure_window)

        with torch.no_grad():
            u_pred = model(attr, adj_matrix)

        anomaly_time_series.append(u_pred.numpy())

    A = np.array(anomaly_time_series)          # [T, N]
    score_per_node = np.sum(np.abs(A), axis=0)

    return score_per_node, idx2node, env.leak_node_names


# ============================================================
# MULTIPLE TESTS
# ============================================================

def run_multiple_tests(
    inp_path,
    model,
    num_test=20,
    max_steps=50,
    window_size=4,
    leak_area=0.1
):
    scores = []

    for i in range(num_test):
        print(f"\n=== TEST {i+1}/{num_test} ===")

        score_per_node, idx2node, leak_nodes = run_single_test_episode(
            inp_path,
            model,
            max_steps,
            window_size,
            leak_area
        )

        loc_score = ranking_score_lexicographic(
            score_per_node,
            idx2node,
            leak_nodes
        )

        scores.append(loc_score)

        print("Leak nodes:", leak_nodes)
        print("Localization score:", loc_score)

    scores = np.array(scores)

    print("\n================= SUMMARY =================")
    print(f"Num test        : {num_test}")
    print(f"Mean score      : {scores.mean():.4f}")
    print(f"Std score       : {scores.std():.4f}")

    return scores


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found/Jilin_copy_copy.inp"
    topo_mlp_ckpt = r"/home/zagaria/Tesi/Tesi/piu-files/saved_models/topo_mlp_ggnn.pt"

    model = load_topo_mlp_model(
        ckpt_path=topo_mlp_ckpt,
        inp_path=inp_path
    )

    run_multiple_tests(
        inp_path=inp_path,
        model=model,
        num_test=30,
        max_steps=50,
        window_size=4,
        leak_area=0.1
    )
