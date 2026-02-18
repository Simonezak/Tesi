import torch
import numpy as np
import random
import networkx as nx

from core.WNTREnv_setup import WNTREnv, build_adj_matrix
from utility.utils_evaluation import leak_detection_error, evaluate_model_across_tests, load_RF, load_CWGGNN

# ============================================================
# PIPELINE RF + CWGGNNN
# ============================================================

def run_RF_CWGGNN_pipeline_tests(inp_path, cwggnn, rf, num_test=100, num_steps=50, leak_area=0.1, threshold=0.15, K=10):

    # RF metrics
    y_true_all = []
    y_pred_all = []
    det_errors = []

    # CW-GGNN metrics
    scores_per_test = []
    leak_nodes_per_test = []

    # episode counters
    num_episodes_with_leak = 0

    env_tmp = WNTREnv(inp_path, num_steps=1)
    adj_matrix, node2idx, idx2node = build_adj_matrix(env_tmp.wn)
    cols = list(node2idx.keys())

    for i in range(num_test):
        print(f"\n=== PIPELINE CW-GGNN TEST {i+1}/{num_test} ===")

        num_leaks = random.randint(0, 2)

        has_true_leak, has_detected_leak, det_error, score_per_node, leak_nodes = run_single_RF_CWGGNN_pipeline_episode(inp_path, cwggnn, rf, adj_matrix, cols, num_steps, leak_area, threshold, num_leaks)

        y_true_all.append(int(has_true_leak))
        y_pred_all.append(int(has_detected_leak))

        if has_true_leak and has_detected_leak:
            det_errors.append(det_error)
            if score_per_node is not None:
                scores_per_test.append(score_per_node)
                leak_nodes_per_test.append(leak_nodes)

    # ============================================================
    # RF METRICS
    # ============================================================
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    num_episodes_with_leak = np.sum(y_true_all == 1)

    TP = np.sum((y_true_all == 1) & (y_pred_all == 1))
    TN = np.sum((y_true_all == 0) & (y_pred_all == 0))
    FP = np.sum((y_true_all == 0) & (y_pred_all == 1))
    FN = np.sum((y_true_all == 1) & (y_pred_all == 0))

    accuracy = (TP + TN) / len(y_true_all)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    mean_det_error = np.mean(det_errors) if len(det_errors) > 0 else None

    # ============================================================
    # CW-GGNN METRICS
    # ============================================================

    cwggnn_metrics = evaluate_model_across_tests(scores_per_test, idx2node, leak_nodes_per_test, K)

    # ============================================================
    # END-TO-END METRICS
    # ============================================================
    pct_all_localized = 100.0 * cwggnn_metrics["num_detected_and_all_localized"] / num_episodes_with_leak
    pct_partial_localized = 100.0 * cwggnn_metrics["num_detected_and_partial_localized"] / num_episodes_with_leak

    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print("\n================= FINAL PIPELINE RESULTS (RF + CW-GGNN) =================")

    print("\n-- Random Forest --")
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")

    print(f"Mean onset error : {mean_det_error:.2f}")

    print("\n-- CW-GGNN (Localization) --")
    print(f"top{K}_all_leaks   : {cwggnn_metrics['topk_all_leaks']:.2f}%")
    print(f"top{K}_single_leak : {cwggnn_metrics['topk_single_leak']:.2f}%")

    print("\n-- End-to-End (Detection + Localization) --")
    print(f"Percentage of leaks DETECTED and LOCALIZED ALL (Top-{K})       : {pct_all_localized:.2f}%")
    print(f"Percentage of leaks DETECTED and LOCALIZED AT LEAST ONE (Top-{K}): {pct_partial_localized:.2f}%")

    return accuracy, precision, recall, mean_det_error, cwggnn_metrics, pct_all_localized, pct_partial_localized


def run_single_RF_CWGGNN_pipeline_episode(inp_path, cwggnn, rf, adj_matrix, cols, num_steps, leak_area, threshold,num_leaks):

    env = WNTREnv(inp_path, num_steps=num_steps)

    env.reset(num_leaks=num_leaks)
    sim = env.sim

    # ------------------------
    # Simulazione WNTR
    # ------------------------
    for step in range(num_steps):
        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(ln, leak_area=leak_area, leak_discharge_coefficient=0.75)
        
        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]

    # ------------------------
    # RF – leak detection
    # ------------------------
    onset_scores = []
    for t in range(len(df_pressure)):
        pressures = df_pressure.iloc[t][cols].to_numpy(np.float32)
        onset_scores.append(rf.predict(pressures))

    onset_scores = np.array(onset_scores)

    has_true_leak = env.leak_start_step is not None
    has_detected_leak = onset_scores.max() >= threshold

    # ------------------------
    # RF
    # ------------------------
    if has_true_leak and has_detected_leak:
        predicted_onset = int(np.argmax(onset_scores))
        det_error = leak_detection_error(predicted_onset, env.leak_start_step)
    else:
        predicted_onset = None
        det_error = None


    # ------------------------
    # CW-GGNN
    # ------------------------
    if has_true_leak and has_detected_leak:

        anomaly_ts = []

        for t in range(predicted_onset, len(df_pressure)):
            attr = torch.tensor(
                df_pressure.iloc[t][cols].to_numpy(np.float32)
            ).unsqueeze(0).unsqueeze(-1)

            with torch.no_grad():
                u_pred = cwggnn(attr, adj_matrix).view(-1)

            anomaly_ts.append(u_pred.cpu().numpy())

        A = np.array(anomaly_ts)
        score_per_node = np.sum(np.abs(A), axis=0)

        return has_true_leak, has_detected_leak, det_error, score_per_node, env.leak_node_names

    return has_true_leak, has_detected_leak, det_error, None, None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"

    cwggnn_path = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/cw_ggnn_GOLD.pt"
    rf_path     = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/rf_leak_onset.pkl"

    NUM_TEST  = 1000
    NUM_STEPS = 50
    THRESHOLD = 0.15
    TOP_K     = 8

    rf = load_RF(rf_path)
    cwggnn = load_CWGGNN(cwggnn_path, inp_path)

    run_RF_CWGGNN_pipeline_tests(inp_path=inp_path, cwggnn=cwggnn, rf=rf, num_test=NUM_TEST, num_steps=NUM_STEPS, threshold=THRESHOLD, K=TOP_K)
