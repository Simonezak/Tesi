import torch
import pickle
import numpy as np

from optuna_GGNN import ranking_score_lexicographic, leak_detection_error

from wntr_exp_multi import (
    WNTREnv,
    build_static_graph_from_wntr
)

# ============================================================
#  IMPORT MODELLO (STESSO DEL TRAINING)
# ============================================================

from prova_LSTM import (
    GGNNEncoder,
    GGNN_NodeLSTM_Localizer
)

def ranking_position_score(ranking_nodes, leak_nodes):
    """
    Score basato sulla posizione finale dei leak nella classifica.
    1° -> 1.0, 2° -> 0.5, k° -> 1/k
    """

    score = 0.0

    for ln in leak_nodes:
        if ln in ranking_nodes:
            rank = ranking_nodes.index(ln)  # 0-based
            score += 1.0 / (rank + 1)

    return score


# ============================================================
#                   LOAD MODELS
# ============================================================

def load_models_embedding_lstm(ckpt_path, rf_model_path, device="cpu"):

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    # ---- GGNN Encoder ----
    ggnn_encoder = GGNNEncoder(
        attr_size=1,
        hidden_size=cfg["ggnn_hidden"],
        propag_steps=cfg["ggnn_propag"]
    ).to(device)

    # ---- Modello completo ----
    model = GGNN_NodeLSTM_Localizer(
        ggnn_encoder=ggnn_encoder,
        hidden_size=cfg["ggnn_hidden"],
        lstm_hidden=cfg["lstm_hidden"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Random Forest (leak onset) ----
    with open(rf_model_path, "rb") as f:
        rf = pickle.load(f)

    print("[OK] Modello GGNN + Node-LSTM caricato correttamente")

    return model, rf


# ============================================================
#                   TEST EPISODE
# ============================================================

def run_single_test_episode_lstm(
    inp_path,
    model,
    rf,
    max_steps,
    leak_area,
    device="cpu"
):
    """
    Test di un singolo episodio:
    - RF per leak onset
    - GGNN + LSTM per localizzazione
    """

    env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(device)

    n_leaks = np.random.randint(1, 3)
    env.reset(num_leaks=n_leaks)
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
    # Leak onset detection (RF)
    # ------------------------
    onset_scores = []

    for t in range(len(df_pressure)):
        pressures = df_pressure.iloc[t][cols].to_numpy(dtype=np.float32)
        onset_scores.append(rf.predict(pressures))

    predicted_onset = int(np.argmax(onset_scores))
    predicted_onset = min(predicted_onset, max_steps - 1)

    true_onset = env.leak_start_step
    det_error = leak_detection_error(predicted_onset, true_onset)

    # ------------------------
    # Leak localization (GGNN + LSTM)
    # ------------------------
    pressure_seq = []
    anomaly_time_series = []

    for t in range(predicted_onset, len(df_pressure)):

        p = torch.tensor(
            df_pressure.iloc[t][cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device
        )

        pressure_seq.append(p)

        # 🔹 tutta la storia fino a t
        attr_seq = [ps.view(1, -1, 1) for ps in pressure_seq]

        with torch.no_grad():
            scores = model(attr_seq, adj_matrix)

        anomaly_time_series.append(scores.cpu().numpy())

    # ------------------------
    # Score per nodo
    # ------------------------
    A = np.array(anomaly_time_series)      # [T, N]
    score_per_node = np.sum(np.abs(A), axis=0)

    ranking_idx = np.argsort(-score_per_node)
    ranking_nodes = [idx2node[i] for i in ranking_idx]


    return score_per_node, idx2node, env.leak_node_names, det_error, ranking_idx, ranking_nodes


# ============================================================
#                   MULTIPLE TESTS
# ============================================================

def run_multiple_tests_lstm(
    inp_path,
    model,
    rf,
    num_test=20,
    max_steps=50,
    leak_area=0.1,
    device="cpu"
):

    localization_scores = []
    detection_errors = []

    for test_id in range(num_test):
        print(f"\n=== TEST {test_id+1}/{num_test} ===")

        score_per_node, idx2node, leak_nodes, det_error, ranking_idx, ranking_nodes = run_single_test_episode_lstm(
            inp_path=inp_path,
            model=model,
            rf=rf,
            max_steps=max_steps,
            leak_area=leak_area,
            device=device
        )

        loc_score = ranking_position_score(
            ranking_nodes,
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

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    inp_path = "/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"

    embedding_lstm_path = (
        "/home/zagaria/Tesi/Tesi/piu-files/saved_models/ggnn_encoder_node_lstm.pt"
    )

    rf_path = (
        "/home/zagaria/Tesi/Tesi/piu-files/saved_models/rf_leak_onset.pkl"
    )

    model, rf = load_models_embedding_lstm(
        embedding_lstm_path,
        rf_path,
        device=DEVICE
    )

    run_multiple_tests_lstm(
        inp_path=inp_path,
        model=model,
        rf=rf,
        num_test=30,
        max_steps=50,
        leak_area=0.1,
        device=DEVICE
    )
