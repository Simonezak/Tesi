import torch
import pickle
import numpy as np
from optuna_GGNN import ranking_score_lexicographic, leak_detection_error

from wntr_exp_Regression import (
    WNTREnv,
    build_static_graph_from_wntr,
    build_attr_from_pressure_window
)
from evaluation import (
    evaluate_model_across_tests_lexicographic
)


from GGNN_Regression import GGNNModel

# ============================================================
#                   PERFORMANCE METRICS
# ============================================================

def hit_at_k(predicted_onsets, true_onsets, k=5):
    predicted_onsets = np.array(predicted_onsets)
    true_onsets = np.array(true_onsets)
    return np.mean(np.abs(predicted_onsets - true_onsets) <= k)

def detection_accuracy(detected, has_leak):
    """
    detected: bool
    has_leak: bool
    """
    return int(detected == has_leak)

def top1_accuracy_multileak(score_per_node, idx2node, leak_nodes):
    ranking = np.argsort(-score_per_node)
    top_L = [idx2node[i] for i in ranking[:len(leak_nodes)]]
    return int(set(top_L) == set(leak_nodes))

def topk_accuracy_multileak(score_per_node, idx2node, leak_nodes, k):
    ranking = np.argsort(-score_per_node)
    top_k = [idx2node[i] for i in ranking[:k]]
    return int(all(ln in top_k for ln in leak_nodes))

def mean_reciprocal_rank(score_per_node, idx2node, leak_nodes):
    ranking = np.argsort(-score_per_node)
    ranking_nodes = [idx2node[i] for i in ranking]

    rr = []
    for ln in leak_nodes:
        rank = ranking_nodes.index(ln) + 1
        rr.append(1.0 / rank)

    return np.mean(rr)

def confidence_ratio(score_per_node, idx2node, leak_nodes, eps=1e-8):
    leak_scores = []
    non_leak_scores = []

    for i, s in enumerate(score_per_node):
        node = idx2node[i]
        if node in leak_nodes:
            leak_scores.append(s)
        else:
            non_leak_scores.append(s)

    return (np.mean(leak_scores) + eps) / (np.mean(non_leak_scores) + eps)

def confidence_gap(score_per_node, idx2node, leak_nodes):
    leak_scores = []
    non_leak_scores = []

    for i, s in enumerate(score_per_node):
        node = idx2node[i]
        if node in leak_nodes:
            leak_scores.append(s)
        else:
            non_leak_scores.append(s)

    return np.mean(leak_scores) - np.mean(non_leak_scores)


# ============================================================
#                   LOAD MODELS
# ============================================================

def load_models(ggnn_ckpt_path,rf_model_path):
    # ---- Load GGNN ----
    ckpt = torch.load(ggnn_ckpt_path, map_location="cpu")

    model = GGNNModel(
        attr_size=ckpt["attr_size"],
        hidden_size=ckpt["hidden_size"],
        propag_steps=ckpt["propag_steps"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to("cpu")
    model.eval()

    # ---- Load Random Forest ----
    with open(rf_model_path, "rb") as f:
        rf = pickle.load(f)

    print("[OK] Modelli caricati correttamente")

    return model, rf


# ============================================================
#                   TEST EPISODE
# ============================================================

def run_single_test_episode(
    inp_path,
    model,
    rf,
    max_steps,
    window_size,
    leak_area
):
    """
    Esegue un singolo episodio di test e restituisce:
    - score_per_node
    - idx2node
    - leak_node_names
    """

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
    # Leak onset detection
    # ------------------------
    onset_scores = []

    for t in range(len(df_pressure)):
        pressures = df_pressure.iloc[t][cols].to_numpy(dtype=np.float32)
        onset_scores.append(rf.predict(pressures))

    predicted_onset = int(np.argmax(onset_scores))

    # per handlare l'errore, da cambiare. l'errore avviene se la detection è troppo tardi e non riesce a formare la finestra
    if predicted_onset > 40:
        predicted_onset = 40


    true_onset = env.leak_start_step

    det_error = leak_detection_error(predicted_onset, true_onset)

    # ------------------------
    # Leak localization (GGNN)
    # ------------------------
    pressure_window = []
    anomaly_time_series = []

    for t in range(predicted_onset, len(df_pressure)):
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
            u_pred = model(attr, adj_matrix).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())

    # ------------------------
    # score_per_node
    # ------------------------
    A = np.array(anomaly_time_series)     # [T, N]
    score_per_node = np.sum(np.abs(A), axis=0)  # [N]

    return score_per_node, idx2node, env.leak_node_names, det_error



def run_multiple_tests(
    inp_path,
    model,
    rf,
    num_test=100,
    max_steps=50,
    window_size=4,
    leak_area=0.1,
    X=2
):
    # ---- per metriche finali lessicografiche
    scores_per_test = []
    leak_nodes_per_test = []

    # ---- metriche già esistenti
    localization_scores = []
    detection_errors = []

    for test_id in range(num_test):
        print(f"\n=== TEST {test_id+1}/{num_test} ===")

        score_per_node, idx2node, leak_nodes, det_error = run_single_test_episode(
            inp_path=inp_path,
            model=model,
            rf=rf,
            max_steps=max_steps,
            window_size=window_size,
            leak_area=leak_area
        )

        # ------------------------
        # metriche CONTINUE
        # ------------------------
        loc_score = ranking_score_lexicographic(
            score_per_node,
            idx2node,
            leak_nodes
        )

        localization_scores.append(loc_score)
        detection_errors.append(det_error)

        # ------------------------
        # metriche LESSICOGRAFICHE (dense rank)
        # ------------------------
        scores_per_test.append(score_per_node)
        leak_nodes_per_test.append(leak_nodes)

        print(f"Leak nodes         : {leak_nodes}")
        print(f"Localization score : {loc_score:.4f}")
        print(f"Detection error    : {det_error}")

    # ====================================================
    # METRICHE FINALI
    # ====================================================

    localization_scores = np.array(localization_scores)
    detection_errors = np.array(detection_errors)

    lex_metrics = evaluate_model_across_tests_lexicographic(
        scores_per_test=scores_per_test,
        idx2node=idx2node,
        leak_nodes_per_test=leak_nodes_per_test,
        X=X
    )

    print("\n================= SUMMARY =================")
    print(f"Num test : {num_test}")

    print("\n--- Localization (continuous score) ---")
    print(f"Mean score             : {localization_scores.mean():.4f}")
    print(f"Std score              : {localization_scores.std():.4f}")

    print("\n--- Detection (RF) ---")
    print(f"Mean detection error   : {detection_errors.mean():.2f}")
    print(f"Mean |detection error| : {np.mean(np.abs(detection_errors)):.2f}")
    print(f"Min detection error    : {detection_errors.min()}")
    print(f"Max detection error    : {detection_errors.max()}")

    print("\n--- Localization (lexicographic, dense rank) ---")
    for k, v in lex_metrics.items():
        print(f"{k:15s}: {v:.2f}%")

    return localization_scores, detection_errors, lex_metrics






# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"

    ggnn_path = r"/home/zagaria/Tesi/Tesi/piu-files/saved_models/ggnn_model.pt"
    rf_path   = r"/home/zagaria/Tesi/Tesi/piu-files/saved_models/rf_leak_onset_a.pkl"

    model, rf = load_models(
        ggnn_ckpt_path=ggnn_path,
        rf_model_path=rf_path
    )

    run_multiple_tests(
        inp_path=inp_path,
        model=model,
        rf=rf,
        num_test=100,
        max_steps=50,
        window_size=4,
        leak_area=0.1,
        X=2
    )

