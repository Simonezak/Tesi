import pickle
import numpy as np
import random

from core.WNTREnv_setup import WNTREnv, build_adj_matrix
from utility.utils_evaluation import leak_detection_error, load_RF

# ============================================================
# RF MODEL TESTING
# ============================================================

def run_RF_test(inp_path, rf, num_test=200, num_steps=50, leak_area=0.1, threshold=0.15):

    y_true_all = []
    y_pred_all = []
    det_errors = []

    for i in range(num_test):
        print(f"\n=== RF TEST {i+1}/{num_test} ===")

        num_leaks = random.randint(0, 3)

        has_true_leak, has_detected_leak, det_error = run_single_RF_test_episode(inp_path, rf, num_steps, leak_area, num_leaks, threshold)

        # --- binary labels ---
        y_true_all.append(has_true_leak)
        y_pred_all.append(has_detected_leak)

        # --- onset error (solo TP) ---
        if has_true_leak == 1 and has_detected_leak == 1 and det_error is not None:
            det_errors.append(det_error)

    # ---- Metriche finali ----
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    TP = np.sum((y_true_all == 1) & (y_pred_all == 1))
    TN = np.sum((y_true_all == 0) & (y_pred_all == 0))
    FP = np.sum((y_true_all == 0) & (y_pred_all == 1))
    FN = np.sum((y_true_all == 1) & (y_pred_all == 0))

    accuracy = (TP + TN) / len(y_true_all)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    mean_det_error = np.mean(det_errors) if len(det_errors) > 0 else None

    # ------------------------
    # Print risultati
    # ------------------------
    print("\n================= RF MODEL RESULTS =================")
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")

    if mean_det_error is not None:
        print(f"\nMean leak detection error : {mean_det_error:.2f}")
    else:
        print("Mean leak detection error : N/A")

    return accuracy, precision, recall, mean_det_error


def run_single_RF_test_episode(inp_path, rf, num_steps, leak_area, num_leaks, threshold):
    
    env = WNTREnv(inp_path, num_steps=num_steps)
    _, node2idx, _ = build_adj_matrix(env.wn)

    env.reset(num_leaks=num_leaks)
    sim = env.sim

    for step in range(num_steps):

        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(ln, leak_area, leak_discharge_coefficient=0.75)
        
        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]
    cols = list(node2idx.keys())

    onset_scores = []
    for t in range(len(df_pressure)):
        pressures = df_pressure.iloc[t][cols].to_numpy(np.float32)
        onset_scores.append(rf.predict(pressures))

    onset_scores = np.array(onset_scores)

    has_true_leak = env.leak_start_step is not None
    has_detected_leak = onset_scores.max() >= threshold

    if has_true_leak and has_detected_leak:
        predicted_onset = int(np.argmax(onset_scores))
        det_error = leak_detection_error(predicted_onset, env.leak_start_step)
    else:
        det_error = None

    return has_true_leak, has_detected_leak, det_error


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    rf_path  = r"/home/zagaria/Tesi/Tesi/wdn-CW-leak-localization/saved_models/rf_leak_onset.pkl"

    NUM_TEST = 100
    NUM_STEPS = 50
    THRESHOLD = 0.15

    rf = load_RF(rf_path)

    run_RF_test(inp_path, rf, NUM_TEST, NUM_STEPS, threshold=THRESHOLD)
