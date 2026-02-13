import torch
import pickle
import numpy as np

from optuna_GGNN import ranking_score_lexicographic, leak_detection_error

from wntr_exp_Regression import (
    WNTREnv,
    build_static_graph_from_wntr
)

# 🔹 IMPORT DEL MODELLO LSTM
from train_model_LSTM import GGNNEmbeddingLSTM
from GGNN_Regression import GGNNModel

# ============================================================
#                   LOAD MODELS
# ============================================================

def load_models_embedding_lstm(ckpt_path, rf_model_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    # ---- GGNN (embedding) ----
    ggnn = GGNNModel(
        attr_size=1,
        hidden_size=cfg["ggnn_hidden"],
        propag_steps=cfg["ggnn_propag"]
    )

    # congelato (come in training)
    for p in ggnn.parameters():
        p.requires_grad = False

    # ---- Modello completo ----
    model = GGNNEmbeddingLSTM(
        ggnn=ggnn,
        lstm_hidden=cfg["lstm_hidden"]
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to("cpu")
    model.eval()

    # ---- RF ----
    import pickle
    with open(rf_model_path, "rb") as f:
        rf = pickle.load(f)

    print("[OK] Modello GGNN-embedding + LSTM caricato")

    return model, rf, cfg["temp_window"]



# ============================================================
#                   TEST EPISODE
# ============================================================

def run_single_test_episode_lstm(
    inp_path,
    model,
    rf,
    max_steps,
    temp_window,
    leak_area
):
    """
    IDENTICO a run_single_test_episode,
    ma usa GGNN + LSTM per la localizzazione
    """

    env = WNTREnv(inp_path, max_steps=max_steps, num_leaks=2)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    env.reset(with_leak=True)
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

    if predicted_onset > 40:
        predicted_onset = 40

    true_onset = env.leak_start_step
    det_error = leak_detection_error(predicted_onset, true_onset)

    # ------------------------
    # Leak localization (GGNN embedding + LSTM)
    # ------------------------
    pressure_window = []
    anomaly_time_series = []

    for t in range(predicted_onset, len(df_pressure)):
        p = torch.tensor(
            df_pressure.iloc[t][cols].to_numpy(dtype=np.float32)
        )
        pressure_window.append(p)

        if len(pressure_window) < temp_window:
            continue
        if len(pressure_window) > temp_window:
            pressure_window.pop(0)

        # costruisci sequenza attr per GGNN
        attr_seq = [pw.view(1, -1, 1) for pw in pressure_window]

        with torch.no_grad():
            u_pred = model(attr_seq, adj_matrix)  # [N]

        anomaly_time_series.append(u_pred.cpu().numpy())

    # ------------------------
    # score_per_node (IDENTICO)
    # ------------------------
    A = np.array(anomaly_time_series)           # [T, N]
    score_per_node = np.sum(np.abs(A), axis=0)  # [N]

    return score_per_node, idx2node, env.leak_node_names, det_error


# ============================================================
#                   MULTIPLE TESTS
# ============================================================

def run_multiple_tests_lstm(
    inp_path,
    model,
    rf,
    temp_window,
    num_test=20,
    max_steps=50,
    leak_area=0.1
):

    localization_scores = []
    detection_errors = []

    for test_id in range(num_test):
        print(f"\n=== TEST {test_id+1}/{num_test} ===")

        score_per_node, idx2node, leak_nodes, det_error = run_single_test_episode_lstm(
            inp_path=inp_path,
            model=model,
            rf=rf,
            max_steps=max_steps,
            temp_window=temp_window,
            leak_area=leak_area
        )

        loc_score = ranking_score_lexicographic(
            score_per_node,
            idx2node,
            leak_nodes
        )

        localization_scores.append(loc_score)
        detection_errors.append(det_error)

        print(f"Leak nodes          : {leak_nodes}")
        print(f"Localization score  : {loc_score:.4f}")
        print(f"Detection error     : {det_error}")

    localization_scores = np.array(localization_scores)
    detection_errors = np.array(detection_errors)

    print("\n================= SUMMARY =================")
    print(f"Num test                 : {num_test}")

    print("\n--- Localization ---")
    print(f"Mean score               : {localization_scores.mean():.4f}")
    print(f"Std score                : {localization_scores.std():.4f}")

    print("\n--- Detection (RF) ---")
    print(f"Mean detection error     : {detection_errors.mean():.2f}")
    print(f"Mean |detection error|   : {np.mean(np.abs(detection_errors)):.2f}")
    print(f"Min detection error      : {detection_errors.min()}")
    print(f"Max detection error      : {detection_errors.max()}")

    return localization_scores, detection_errors


# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched.inp"

    embedding_lstm_path = (r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\saved_models\ggnn_embedding_lstm.pt")
    rf_path   = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\saved_models\rf_leak_onset.pkl"

    model, rf, temp_window = load_models_embedding_lstm(embedding_lstm_path,rf_path)

    run_multiple_tests_lstm(
        inp_path=inp_path,
        model=model,
        rf=rf,
        temp_window=temp_window,
        num_test=30,
        max_steps=50,
        leak_area=0.1
    )
