import torch
import numpy as np
import random

from core.WNTREnv_setup import WNTREnv, build_adj_matrix
from utility.utils_evaluation import evaluate_model_across_tests, load_GGNN

# ============================================================
# GGNN MODEL TESTING
# ============================================================

def run_GGNN_test(inp_path, model, num_test=100, num_steps=50, leak_area=0.1, K=10):

    scores_per_test = []
    leak_nodes_per_test = []

    for i in range(num_test):
        print(f"\n=== GGNN TEST {i+1}/{num_test} ===")

        #num_leaks = random.randint(1, 2)
        num_leaks = 2

        score_per_node, idx2node, leak_nodes = run_single_GGNN_test_episode(inp_path, model, num_steps, leak_area, num_leaks)

        scores_per_test.append(score_per_node)
        leak_nodes_per_test.append(leak_nodes)

    # Ranking metrics
    metrics = evaluate_model_across_tests(scores_per_test, idx2node, leak_nodes_per_test, K)

    print("\n================= GGNN RESULTS =================")
    print(f"top{K}_all_leaks   : {metrics['topk_all_leaks']:.2f}%")
    print(f"top{K}_single_leak : {metrics['topk_single_leak']:.2f}%")

    return metrics


def run_single_GGNN_test_episode(inp_path, model, num_steps, leak_area, num_leaks):

    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, idx2node = build_adj_matrix(env.wn)

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

    anomaly_ts = []

    for t in range(env.leak_start_step, len(df_pressure)):
        attr = torch.tensor(
            df_pressure.iloc[t][cols].to_numpy(np.float32)
        ).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            u_pred = model(attr, adj_matrix).view(-1)

        anomaly_ts.append(u_pred.cpu().numpy())

    A = np.array(anomaly_ts)
    score_per_node = np.sum(np.abs(A), axis=0)

    return score_per_node, idx2node, env.leak_node_names


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    ggnn_path = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/ggnn_model.pt"

    NUM_TEST  = 100
    NUM_STEPS = 50
    TOP_K     = 5

    model = load_GGNN(ggnn_path)

    run_GGNN_test(inp_path=inp_path, model=model, num_test=NUM_TEST, num_steps=NUM_STEPS, K=TOP_K)
