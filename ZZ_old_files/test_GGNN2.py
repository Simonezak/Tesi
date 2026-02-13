import torch
import numpy as np
import random
import networkx as nx

from models.GGNN import GGNNModel
from core.WNTREnv_setup import (
    WNTREnv,
    build_adj_matrix,
    build_edge2pipe_mapping
)
from utility.utils_evaluation import evaluate_model_across_tests
from utility.utils_cellcomplex import func_gen_B2_lu


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PATH
# ============================================================
inp_path = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched.inp"
model_ckpt = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\saved_models\train_ggnn.pt"


# ============================================================
# TEST CONFIG (IDENTICO AL REPO)
# ============================================================
NUM_TESTS = 10
num_steps = 50
LEAK_AREA = 0.1
num_leaks = 3


# ============================================================
# INPUT CONFIG (NUOVO SCHEMA)
# ============================================================
use_pressure = True
use_flow     = False

pressure_keep_frac = 1.0
flow_keep_frac     = 0.0


# ============================================================
# SEED (RIPRODUCIBILITÀ)
# ============================================================
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# UTILS
# ============================================================
def sample_binary_mask(n, keep_frac):
    if keep_frac >= 1.0:
        return torch.ones(n, device=DEVICE)
    if keep_frac <= 0.0:
        return torch.zeros(n, device=DEVICE)

    k = max(1, int(round(keep_frac * n)))
    idx = torch.randperm(n, device=DEVICE)[:k]
    m = torch.zeros(n, device=DEVICE)
    m[idx] = 1.0
    return m


def load_GGNN_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = GGNNModel(
        attr_size=ckpt["attr_size"],
        hidden_size=ckpt["hidden_size"],
        propag_steps=ckpt["propag_steps"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ============================================================
# TEST (STRUTTURA IDENTICA A test_GGNN DEL REPO)
# ============================================================
def test_GGNN():

    assert use_pressure or use_flow, "At least one signal must be enabled."

    model = load_GGNN_model(model_ckpt)
    attr_size = model.attr_size

    env = WNTREnv(inp_path, num_steps=num_steps, seed=seed)
    adj_matrix, node2idx, idx2node = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    cols = list(node2idx.keys())
    N = len(cols)

    # -------- flow structure (NUOVO) --------
    if use_flow:
        edge2pipe = build_edge2pipe_mapping(env.wn, node2idx)
        edge_list = sorted(edge2pipe.keys())

    all_predictions = []
    all_true_leaks = []

    print("\n=== TEST GGNN (schema P/F + mask) ===")
    print(f"attr_size = {attr_size}\n")

    # ========================================================
    # LOOP TEST (IDENTICO)
    # ========================================================
    for test_id in range(NUM_TESTS):

        env.reset(num_leaks)
        sim = env.sim

        for step in range(num_steps):
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=LEAK_AREA)
            sim.step_sim()

        results = sim.get_results()

        if use_pressure:
            df_pressure = results.node["pressure"]

        if use_flow:
            df_flow = results.link["flowrate"]

            G = nx.Graph()
            for pipe_name in env.wn.pipe_name_list:
                pipe = env.wn.get_link(pipe_name)
                u = pipe.start_node_name
                v = pipe.end_node_name
                if u in node2idx and v in node2idx:
                    G.add_edge(node2idx[u], node2idx[v])

            B1, B2, _ = func_gen_B2_lu(G, 8)
            B1 = torch.tensor(B1, dtype=torch.float32, device=DEVICE)

        # -------- masks (NUOVO) --------
        if use_pressure:
            p_mask = sample_binary_mask(N, pressure_keep_frac)

        if use_flow:
            e_mask = sample_binary_mask(len(edge_list), flow_keep_frac)

        # === IDENTICO AL REPO: ultimo timestep ===
        t = num_steps - 1

        features = []

        # ---------------- pressure ----------------
        if use_pressure:
            pressure = torch.tensor(
                df_pressure.iloc[t][cols].values,
                dtype=torch.float32,
                device=DEVICE,
            )
            p_obs = pressure * p_mask
            features.append(p_obs)
            features.append(p_mask)

        # ---------------- flow ----------------
        if use_flow:
            # q sugli archi (con segno)
            q = torch.zeros(len(edge_list), device=DEVICE)
            row = df_flow.iloc[t]
            for e_idx, (u, v) in enumerate(edge_list):
                pipe_name = edge2pipe[(u, v)]
                q[e_idx] = float(row[pipe_name])

            # maschera archi osservati
            q_obs = q * e_mask

            # === NUOVA FEATURE: bilancio nodale ===
            flow_balance = B1 @ q_obs           # [N]

            # maschera nodale: almeno un arco osservato
            flow_mask_node = ((B1.abs() @ e_mask) > 0).float()

            features.append(flow_balance)
            features.append(flow_mask_node)

        X = torch.stack(features, dim=-1)

        # -------- inference (IDENTICO) --------
        with torch.no_grad():
            out = model(X, adj_matrix)

        all_predictions.append(out.detach().cpu().numpy())
        all_true_leaks.append(env.leak_node_names)

        print(f"Test {test_id+1}/{NUM_TESTS} done")

    # ========================================================
    # EVALUATION (IDENTICA AL REPO)
    # ========================================================
    
    metrics = evaluate_model_across_tests(all_predictions, idx2node, all_true_leaks, num_leaks)

    K2 = 5

    metrics2 = evaluate_model_across_tests(all_predictions, idx2node, all_true_leaks, K2)

    K3 = 8

    metrics3 = evaluate_model_across_tests(all_predictions, idx2node, all_true_leaks, K3)

    K4 = 10

    metrics4 = evaluate_model_across_tests(all_predictions, idx2node, all_true_leaks, K4)

    print("\n================= CW-GGNN RESULTS =================")
    print(f"top{num_leaks}_all_leaks   : {metrics['topk_all_leaks']:.2f}%")
    print(f"top{num_leaks}_single_leak : {metrics['topk_single_leak']:.2f}%")

    print("\n================= CW-GGNN RESULTS =================")
    print(f"top{K2}_all_leaks   : {metrics2['topk_all_leaks']:.2f}%")
    print(f"top{K2}_single_leak : {metrics2['topk_single_leak']:.2f}%")

    print("\n================= CW-GGNN RESULTS =================")
    print(f"top{K3}_all_leaks   : {metrics3['topk_all_leaks']:.2f}%")
    print(f"top{K3}_single_leak : {metrics3['topk_single_leak']:.2f}%")

    print("\n================= CW-GGNN RESULTS =================")
    print(f"top{K4}_all_leaks   : {metrics4['topk_all_leaks']:.2f}%")
    print(f"top{K4}_single_leak : {metrics4['topk_single_leak']:.2f}%")

    return metrics


if __name__ == "__main__":
    test_GGNN()
