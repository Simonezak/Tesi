import torch
import numpy as np
import random
import networkx as nx

from models.GGNN import GGNNModel
from models.CW_GGNN import TopoCycleResidualNodeAlpha, GGNNWithTopoAlpha
from core.WNTREnv_setup import WNTREnv, build_adj_matrix, build_edge2pipe_mapping
from utility.utils_cellcomplex import func_gen_B2_lu
from utility.utils_evaluation import evaluate_model_across_tests

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PATH
# ============================================================
inp_path = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched.inp"
model_ckpt = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\saved_models\train_cwggnn.pt"


# ============================================================
# TEST CONFIG
# ============================================================
NUM_TESTS = 100
num_steps = 50
LEAK_AREA = 0.1
num_leaks = 3

# maschere (coerenti con train nuovo)
pressure_keep_frac = 1.0
flow_keep_frac     = 0.0


# ============================================================
# SEED
# ============================================================
seed = 52
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


def _build_topology_and_edge_order(env, node2idx, max_cycle_length):
    """
    Ricostruisce G in modo coerente col training:
    - inserisce gli archi seguendo wn.pipe_name_list
    - edge_list = list(G.edges()) (in insertion order)
    - B1, B2 coerenti con quell'edge order
    """
    G = nx.Graph()
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        u = pipe.start_node_name
        v = pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    B1_np, B2_np, _ = func_gen_B2_lu(G, max_cycle_length)
    edge_list = list(G.edges())  # ORDER CRITICO: non ordinare/sortare

    return G, edge_list, B1_np, B2_np


def _pipe_name_for_edge(edge2pipe, u, v):
    # robusto rispetto a (u,v)/(v,u) e tuple ordinate
    if (u, v) in edge2pipe:
        return edge2pipe[(u, v)]
    if (v, u) in edge2pipe:
        return edge2pipe[(v, u)]
    a, b = (u, v) if u <= v else (v, u)
    return edge2pipe.get((a, b), None)


def _build_X_from_ckpt(
    ckpt,
    t,
    cols,
    df_pressure,
    df_flow,
    edge2pipe,
    edge_list,
    B1,
    pressure_keep_frac=1.0,
    flow_keep_frac=0.0,
):
    """
    Costruisce X [1,N,F] seguendo ckpt["feature_names"] (o fallback).
    Feature supportate (coerenti col train nuovo proposto):
      - "p_obs"
      - "p_mask"
      - "flow_balance"
      - "flow_mask_node"
      - "flow_imbalance"
    """
    feature_names = tuple(ckpt.get("feature_names", ("p_obs", "p_mask", "flow_balance", "flow_mask_node")))
    N = len(cols)
    E = len(edge_list)

    p_mask = sample_binary_mask(N, pressure_keep_frac)
    e_mask = sample_binary_mask(E, flow_keep_frac)

    # ---- pressure ----
    pressure = torch.tensor(df_pressure.iloc[t][cols].values, dtype=torch.float32, device=DEVICE)
    p_obs = pressure * p_mask

    # ---- flow ----
    # se manca df_flow, metti zero (ma per test serio meglio che ci sia)
    q = torch.zeros(E, device=DEVICE)
    if df_flow is not None:
        row = df_flow.iloc[t]
        for e_idx, (u, v) in enumerate(edge_list):
            pn = _pipe_name_for_edge(edge2pipe, u, v)
            q[e_idx] = 0.0 if pn is None else float(row[pn])

    q_obs = q * e_mask
    flow_balance = B1 @ q_obs  # [N]
    flow_mask_node = ((B1.abs() @ e_mask) > 0).float()
    flow_imbalance = flow_balance.abs()

    feat_map = {
        "p_obs": p_obs,
        "p_mask": p_mask,
        "flow_balance": flow_balance,
        "flow_mask_node": flow_mask_node,
        "flow_imbalance": flow_imbalance,
    }

    feats = [feat_map[name] for name in feature_names]
    print(feats)
    X = torch.stack(feats, dim=-1).unsqueeze(0)  # [1,N,F]
    print("ciao")
    print(X)
    return X


def load_CWGGNN_model(ckpt_path, env, node2idx):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # ===== topology coerente con ckpt =====
    max_cycle_length = ckpt.get("max_cycle_length", 8)
    _, _, B1_np, B2_np = _build_topology_and_edge_order(env, node2idx, max_cycle_length)

    # ===== GGNN =====
    ggnn = GGNNModel(
        attr_size=int(ckpt.get("attr_size", 1)),
        hidden_size=int(ckpt["hidden_size"]),
        propag_steps=int(ckpt["propag_steps"]),
    ).to(DEVICE)

    topo_node = TopoCycleResidualNodeAlpha(
        B1_np=B1_np,
        B2_np=B2_np,
        hidden_dim=int(ckpt["hidden_size"]),
        alpha_max=0.1,
    ).to(DEVICE)

    model = GGNNWithTopoAlpha(ggnn, topo_node).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


# ============================================================
# TEST
# ============================================================
def test_CWGGNN():

    env = WNTREnv(inp_path, num_steps=num_steps, seed=seed)
    adj_matrix, node2idx, idx2node = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    cols = list(node2idx.keys())

    # per leggere flowrate (pipe name) dagli edge
    edge2pipe = build_edge2pipe_mapping(env.wn, node2idx)

    # topology + edge order + B1 coerenti con train/ckpt
    ckpt_tmp = torch.load(model_ckpt, map_location="cpu")
    max_cycle_length = ckpt_tmp.get("max_cycle_length", 8)
    _, edge_list, B1_np, _ = _build_topology_and_edge_order(env, node2idx, max_cycle_length)
    B1 = torch.tensor(B1_np, dtype=torch.float32, device=DEVICE)

    model, ckpt = load_CWGGNN_model(model_ckpt, env, node2idx)
    #----
    all_predictions = []
    all_true_leaks = []

    print("\n=== TEST CW-GGNN (MULTI-INPUT) ===\n")
    print(f"ckpt feature_names = {ckpt.get('feature_names', None)}")
    print(f"ckpt attr_size      = {ckpt.get('attr_size', None)}\n")

    for test_id in range(NUM_TESTS):

        env.reset(num_leaks)
        sim = env.sim

        for step in range(num_steps):
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=LEAK_AREA)
            sim.step_sim()

        results = sim.get_results()

        df_pressure = results.node["pressure"]
        df_flow = results.link.get("flowrate", None)

        # (come nel tuo base) ultimo timestep
        t = num_steps - 1

        X = _build_X_from_ckpt(
            ckpt=ckpt,
            t=t,
            cols=cols,
            df_pressure=df_pressure,
            df_flow=df_flow,
            edge2pipe=edge2pipe,
            edge_list=edge_list,
            B1=B1,
            pressure_keep_frac=pressure_keep_frac,
            flow_keep_frac=flow_keep_frac,
        )

        with torch.no_grad():
            out = model(X, adj_matrix)  # [N]

        all_predictions.append(out.cpu().numpy())
        all_true_leaks.append(env.leak_node_names)

        print(f"Test {test_id+1}/{NUM_TESTS} done")

    metrics = evaluate_model_across_tests(
        all_predictions,
        idx2node,
        all_true_leaks,
        num_leaks,
    )

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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    test_CWGGNN()
