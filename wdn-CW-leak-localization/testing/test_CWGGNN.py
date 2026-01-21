import torch
import numpy as np
import random

from core.WNTREnv_setup import WNTREnv, build_adj_matrix
from utility.utils_evaluation import evaluate_model_across_tests, load_CWGGNN

# ============================================================
# CWGGNN MODEL TESTING
# ============================================================

def run_CWGGNN_test(inp_path, model, num_test=100, num_steps=50, leak_area=0.1, K=10):

    scores_per_test = []
    leak_nodes_per_test = []

    env_tmp = WNTREnv(inp_path, num_steps=1)
    adj_matrix, node2idx, idx2node = build_adj_matrix(env_tmp.wn)

    for i in range(num_test):
        print(f"\n=== TEST {i+1}/{num_test} ===")

        #num_leaks = random.randint(1, 3)
        num_leaks = 3

        score_per_node, leak_nodes = run_single_CWGGNN_test_episode(inp_path, model, num_steps, leak_area, num_leaks, adj_matrix, node2idx)

        scores_per_test.append(score_per_node)
        leak_nodes_per_test.append(leak_nodes)

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


    return metrics


def run_single_CWGGNN_test_episode(inp_path, model, num_steps, leak_area, num_leaks, adj_matrix, node2idx):

    env = WNTREnv(inp_path, num_steps=num_steps)
    
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
        attr = torch.tensor(df_pressure.iloc[t][cols].to_numpy(np.float32)).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            u_pred = model(attr, adj_matrix)

        anomaly_ts.append(u_pred.numpy())

    A = np.array(anomaly_ts)
    score_per_node = np.sum(np.abs(A), axis=0)

    return score_per_node, env.leak_node_names


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found-final/modena_BSD.inp"
    topo_mlp_ckpt = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/cw_ggnn_Modena_GOLD.pt"

    NUM_TEST = 100
    TOP_K = 3

    model = load_CWGGNN(topo_mlp_ckpt, inp_path)

    run_CWGGNN_test(inp_path=inp_path, model=model, num_test=NUM_TEST, K=TOP_K)

