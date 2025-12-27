# optuna.py
import optuna
import numpy as np
import torch
import torch.nn as nn

# ===============================
# IMPORT DAL TUO PROGETTO
# ===============================
from wntr_exp_multi import (
    WNTREnv,
    GGNNModel,
    build_attr_from_pressure_window,
    build_static_graph_from_wntr
)

# ===============================
# CONFIGURAZIONE BASE
# ===============================
INP_PATH = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched.inp"

MAX_STEPS = 50
AREA = 0.1
NUM_EPISODES = 20        # pochi → Optuna deve essere veloce
EPOCHS = 400             # training corto

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# METRICA DI RANKING
# ===============================
def ranking_score(score_per_node, idx2node, leak_nodes):
    """
    Score alto se i nodi leak sono in cima al ranking.
    Usa Reciprocal Rank medio.
    """
    ranking_idx = np.argsort(-score_per_node)
    ranking_nodes = [idx2node[i] for i in ranking_idx]

    score = 0.0
    for ln in leak_nodes:
        if ln in ranking_nodes:
            rank = ranking_nodes.index(ln)
            score += 1.0 / (rank + 1)

    return score

def dense_rank(scores, descending=True):
    """
    Assegna un dense rank ai valori di scores.
    Rank = 1 è il migliore.
    """
    scores = np.asarray(scores)

    # valori unici ordinati
    unique_scores = np.unique(scores)
    if descending:
        unique_scores = unique_scores[::-1]

    # mappa score -> rank
    score_to_rank = {s: r + 1 for r, s in enumerate(unique_scores)}

    # assegna rank a ciascun nodo
    ranks = np.array([score_to_rank[s] for s in scores])

    return ranks

def ranking_score_lexicographic(score_per_node, idx2node, leak_nodes):

    # dense rank
    ranks = dense_rank(score_per_node)

    score_primary = 0.0
    score_secondary = 0.0

    for ln in leak_nodes:
        leak_idx = [i for i, n in idx2node.items() if n == ln][0]

        # score primario: posizione
        score_primary += 1.0 / ranks[leak_idx]

        # score secondario: confidenza
        score_secondary += score_per_node[leak_idx]

    # combinazione lessicografica
    final_score = score_primary + 1e-3 * score_secondary

    return final_score


def ranking_score_lexicographic_no_dense(
    score_per_node,
    idx2node,
    leak_nodes
):
    """
    Score lessicografico:
    - primario: posizione nel ranking classico
    - secondario: intensità anomalia (confidence)
    """

    # ranking classico
    ranking_idx = np.argsort(-score_per_node)
    ranking_nodes = [idx2node[i] for i in ranking_idx]

    score_primary = 0.0
    score_secondary = 0.0

    for ln in leak_nodes:
        # indice numerico del nodo leak
        leak_idx = [i for i, n in idx2node.items() if n == ln][0]

        # posizione nel ranking (0 = primo)
        rank = ranking_nodes.index(ln)

        # 1️⃣ score primario: posizione
        score_primary += 1.0 / (rank + 1)

        # 2️⃣ score secondario: confidenza
        score_secondary += score_per_node[leak_idx]

    # combinazione lessicografica
    final_score = score_primary + 1e-3 * score_secondary

    return final_score

def leak_detection_error(predicted_onset, true_onset):
    """
    Errore di detection del leak.

    < 0 : stimato prima del leak reale
    = 0 : stimato correttamente
    > 0 : stimato dopo il leak reale
    """
    return int(predicted_onset) - int(true_onset)



# ===============================
# FUNZIONE OBIETTIVO OPTUNA
# ===============================
def objective(trial):

    # 🔹 IPERPARAMETRI
    hidden_size = trial.suggest_int("hidden_size", 90, 256)
    propag_steps = trial.suggest_int("propag_steps", 2, 7)
    window_size = trial.suggest_int("window_size", 1, 5)
    lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)

    # ===============================
    # TRAINING
    # ===============================
    env = WNTREnv(INP_PATH, max_steps=MAX_STEPS)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    model = GGNNModel(
        attr_size=window_size,
        hidden_size=hidden_size,
        propag_steps=propag_steps
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for ep in range(NUM_EPISODES):
        env.reset(with_leak=True)
        sim = env.sim

        pressure_window = []
        train_samples = []

        # ---- Simulazione WNTR
        for step in range(MAX_STEPS):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=AREA)

            sim.step_sim()
            results = sim.get_results()

            pressures = torch.tensor(
                results.node["pressure"].iloc[-1][list(node2idx.keys())].values,
                dtype=torch.float32
            )
            pressure_window.append(pressures)

            if len(pressure_window) > window_size:
                pressure_window.pop(0)
            if len(pressure_window) < window_size:
                continue
            if step < env.leak_start_step:
                continue

            attr_matrix = build_attr_from_pressure_window(pressure_window).to(DEVICE)

            demand = results.node["demand"].iloc[-1][list(node2idx.keys())].values
            leak = results.node.get("leak_demand", None)
            leak = leak.iloc[-1][list(node2idx.keys())].values if leak is not None else np.zeros_like(demand)

            target = torch.tensor(demand + leak, dtype=torch.float32).to(DEVICE)
            train_samples.append((attr_matrix, target))

        # ---- Training su pochi sample
        for attr, y in train_samples:
            optimizer.zero_grad()
            out = model(attr, adj_matrix.to(DEVICE)).view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    # ===============================
    # TEST
    # ===============================
    env.reset(with_leak=True)
    sim = env.sim

    pressure_window = []
    anomaly_time_series = []

    for step in range(MAX_STEPS):

        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(ln, leak_area=AREA)

        sim.step_sim()
        results = sim.get_results()

        pressures = torch.tensor(
            results.node["pressure"].iloc[-1][list(node2idx.keys())].values,
            dtype=torch.float32
        )
        pressure_window.append(pressures)

        if len(pressure_window) > window_size:
            pressure_window.pop(0)
        if len(pressure_window) < window_size:
            continue
        if step < env.leak_start_step:
            continue

        attr_matrix = build_attr_from_pressure_window(pressure_window).to(DEVICE)

        with torch.no_grad():
            u_pred = model(attr_matrix, adj_matrix.to(DEVICE)).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())

    A = np.array(anomaly_time_series)        # [T, N]
    score_per_node = A.sum(axis=0)

    score = ranking_score_lexicographic(
        score_per_node,
        idx2node,
        env.leak_node_names
    )

    return score


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="GGNN_WNTR_noSoftplus",
        storage="sqlite:///optuna_ggnn_noSoftpluso.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=30)

    print("\n=== OPTUNA FINISHED ===")
    print("Best score:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open("optuna_results.txt", "w") as f:
        f.write(f"Best score: {study.best_value}\n")
        f.write("Best params:\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")

