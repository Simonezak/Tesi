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
from evaluation import evaluate_model_across_tests_lexicographic


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
        topo_feat = self.L1 @ X
        return X + self.mlp(topo_feat)


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
        u_node = self.ggnn(attr_matrix, adj_matrix).view(-1)
        edge_feat = (self.B1_T @ u_node).unsqueeze(-1)
        edge_feat = self.topo(edge_feat)
        return self.B1 @ edge_feat.squeeze(-1)


# ============================================================
# LOAD MODEL
# ============================================================

def load_topo_mlp_model(ckpt_path, inp_path):

    WINDOW_SIZE = 1
    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6
    TOPO_MLP_HIDDEN = 16

    ggnn = GGNNModel(
        attr_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    )

    env_tmp = WNTREnv(inp_path, max_steps=10)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env_tmp.wn)

    G = nx.Graph()
    for pipe_name in env_tmp.wn.pipe_name_list:
        pipe = env_tmp.wn.get_link(pipe_name)
        u, v = pipe.start_node_name, pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    B1, B2, _ = func_gen_B2_lu(G, max_cycle_length=8)
    L1 = B1.T @ B1 + B2 @ B2.T

    topo_layer = TopoResidual(L1, hidden=TOPO_MLP_HIDDEN)
    model = GGNNWithTopoMLP(ggnn, topo_layer, B1)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    print("[OK] Topo-MLP-GGNN caricato")

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

    for step in range(max_steps):
        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(ln, leak_area=leak_area, leak_discharge_coefficient=0.75)
        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]
    cols = list(node2idx.keys())

    pressure_window = []
    anomaly_ts = []

    for t in range(env.leak_start_step, len(df_pressure)):
        p = torch.tensor(df_pressure.iloc[t][cols].to_numpy(np.float32))
        pressure_window.append(p)

        if len(pressure_window) < window_size:
            continue
        if len(pressure_window) > window_size:
            pressure_window.pop(0)

        attr = build_attr_from_pressure_window(pressure_window)

        with torch.no_grad():
            u_pred = model(attr, adj_matrix)

        anomaly_ts.append(u_pred.numpy())

    A = np.array(anomaly_ts)
    score_per_node = np.sum(np.abs(A), axis=0)

    return score_per_node, idx2node, env.leak_node_names


# ============================================================
# MULTI TEST (COME topo_prova2.py)
# ============================================================

def run_multiple_tests(
    inp_path,
    model,
    num_test=100,
    max_steps=50,
    window_size=1,
    leak_area=0.1,
    X=2
):
    scores_per_test = []
    leak_nodes_per_test = []

    for i in range(num_test):
        print(f"\n=== TEST {i+1}/{num_test} ===")

        score_per_node, idx2node, leak_nodes = run_single_test_episode(
            inp_path, model, max_steps, window_size, leak_area
        )

        scores_per_test.append(score_per_node)
        leak_nodes_per_test.append(leak_nodes)

        print("Leak nodes:", leak_nodes)

    metrics = evaluate_model_across_tests_lexicographic(
        scores_per_test=scores_per_test,
        idx2node=idx2node,
        leak_nodes_per_test=leak_nodes_per_test,
        X=X
    )

    print("\n================= FINAL RESULTS =================")
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.2f}%")

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = "/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    topo_mlp_ckpt = "/home/zagaria/Tesi/Tesi/piu-files/saved_models/topo_mlp_ggnn.pt"

    model = load_topo_mlp_model(topo_mlp_ckpt, inp_path)

    run_multiple_tests(
        inp_path=inp_path,
        model=model,
        num_test=100,
        max_steps=50,
        window_size=1,
        leak_area=0.1,
        X=2
    )
