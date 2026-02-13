import torch
import numpy as np
import random
import networkx as nx

from core.WNTREnv_setup import WNTREnv, build_adj_matrix, sample_binary_mask, build_edge2pipe_mapping
from utility.utils_evaluation import evaluate_model_across_tests, load_CWGGNN, compute_mrr, compute_f1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# CWGGNN MODEL TESTING
# ============================================================

def run_CWGGNN_test(inp_path, num_test=100, num_steps=50, num_leaks=3, leak_area=0.1, use_pressure=True, use_flow=False, pressure_keep_frac=1.0, flow_keep_frac=1.0):
    
    model, edge_list, B1_np, _ = load_CWGGNN(topo_mlp_ckpt, inp_path)

    env = WNTREnv(inp_path, num_steps=num_steps, seed=seed)
    adj_matrix, node2idx, idx2node = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    edge2pipe = build_edge2pipe_mapping(env.wn, node2idx)

    cols = list(node2idx.keys())
    N = len(cols)

    G = nx.Graph()
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        u, v = pipe.start_node_name, pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])
    
    edge_list = list(G.edges())
    E = len(edge_list)

    B1 = torch.tensor(B1_np, dtype=torch.float32, device=DEVICE)

    scores_per_test = []
    leak_nodes_per_test = []

    for i in range(num_test):
        print(f"\n=== TEST {i+1}/{num_test} ===")

        env.reset(num_leaks=num_leaks)
        sim = env.sim

        for step in range(num_steps):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=leak_area, leak_discharge_coefficient=0.75)

            sim.step_sim()

        results = sim.get_results()
        df_pressure = results.node["pressure"]

        cols = list(node2idx.keys())
        N = len(cols)
        E = len(edge_list)

        if use_pressure:
            df_pressure = results.node["pressure"]
            p_mask = sample_binary_mask(N, pressure_keep_frac)

        if use_flow:
            df_flow = results.link["flowrate"]
            e_mask = sample_binary_mask(E, flow_keep_frac)

        t = num_steps - 1
        features = []

        # ---------------- pressure ----------------
        if use_pressure:
            pressure = torch.tensor(df_pressure.iloc[t][cols].values, dtype=torch.float32, device=DEVICE, )
            p_obs = pressure * p_mask
            features.append(p_obs)
            features.append(p_mask)

        # ---------------- flow ----------------
        if use_flow:
            q = torch.zeros(E, device=DEVICE)
            row = df_flow.iloc[t]
            
            for e_idx, (u, v) in enumerate(edge_list):
                pipe_name = edge2pipe.get((u, v)) or edge2pipe.get((v, u))
                if pipe_name is not None:
                    q[e_idx] = float(row[pipe_name])

            q_obs = q * e_mask
            flow_balance = B1 @ q_obs
            flow_mask_node = ((B1.abs() @ e_mask) > 0).float()

            deg = B1.abs().sum(dim=1) + 1e-6
            flow_balance = flow_balance / deg
            flow_balance = flow_balance - flow_balance.mean()

            features.append(flow_balance)
            features.append(flow_mask_node)

        attr = torch.stack(features, dim=-1).unsqueeze(0) 

        with torch.no_grad():
            u_pred = model(attr, adj_matrix)

        scores_per_test.append(u_pred.cpu().numpy())
        leak_nodes_per_test.append(env.leak_node_names)

    metrics = evaluate_model_across_tests(scores_per_test, idx2node, leak_nodes_per_test, num_leaks)

    K2 = 5
    metrics2 = evaluate_model_across_tests(scores_per_test, idx2node, leak_nodes_per_test, K2)

    K3 = 8
    metrics3 = evaluate_model_across_tests(scores_per_test, idx2node, leak_nodes_per_test, K3)

    K4 = 10
    metrics4 = evaluate_model_across_tests(scores_per_test, idx2node, leak_nodes_per_test, K4)

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

    mrr = compute_mrr(scores_per_test, idx2node, leak_nodes_per_test)

    f1_metrics = compute_f1(scores_per_test, idx2node, leak_nodes_per_test, num_leaks)

    print("\n================= MEAN RECIPROCAL RANK =================")
    print(f"MRR               : {mrr:.3f}")
    
    print("\n================= MACRO F1 =================")
    print(f"F1                : {f1_metrics['f1']:.3f}")

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched_BSD.inp"
    topo_mlp_ckpt = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\saved_models\train_cwggnn.pt"

    use_pressure = True
    use_flow = True

    pressure_keep_frac = 1.0
    flow_keep_frac     = 0.0

    NUM_TEST = 50
    num_leaks = 3

    run_CWGGNN_test(inp_path=inp_path, num_test=NUM_TEST, num_leaks=num_leaks, use_pressure=use_pressure, use_flow=use_flow, pressure_keep_frac = pressure_keep_frac, flow_keep_frac=flow_keep_frac)

