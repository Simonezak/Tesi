import optuna
import numpy as np
import torch
import torch.nn as nn
import wntr
import random

from optuna_GGNN import ranking_score_lexicographic
from main_dyn_topologyknown_01 import func_gen_B2_lu

# importa modello e helper dal file di training
from ggnn_cellcomplex_train import (
    GGNNCellComplexModel,
    compute_B1_B2_for_wn,
)

# -------------------------
# Simulator import
# -------------------------
try:
    from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
except Exception:
    from interactive_network_simulator import InteractiveWNTRSimulator


# ============================================================
#                   CONFIGURAZIONE BASE
# ============================================================

INP_PATH = "/home/zagaria/Tesi/Tesi/Networks-found/Jilin.inp"

MAX_STEPS = 50
AREA = 0.1
NUM_EPISODES = 15        # pochi → Optuna deve essere veloce
EPOCHS = 300             # training corto

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#                   WNTR ENV
# ============================================================

class WNTREnv:
    def __init__(self, inp_path, max_steps=50, hydraulic_timestep=3600, num_leaks=2):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.num_leaks = num_leaks
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = None
        self.leak_node_names = []
        self.leak_start_step = None

    def reset(self):
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

        junctions = [
            name for name, node in self.wn.nodes()
            if isinstance(node, wntr.network.elements.Junction)
        ]

        self.leak_node_names = np.random.choice(
            junctions,
            size=min(self.num_leaks, len(junctions)),
            replace=False
        ).tolist()

        self.leak_start_step = int(np.random.randint(10, 26))

        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )


# ============================================================
#                   GRAPH HELPERS
# ============================================================

def build_static_graph_from_wntr(wn):
    node_names = [name for name, _ in wn.nodes()]
    node2idx = {n: i for i, n in enumerate(node_names)}
    idx2node = {i: n for n, i in node2idx.items()}

    N = len(node_names)
    adj = torch.zeros((1, N, N), dtype=torch.float32)

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        u, v = pipe.start_node_name, pipe.end_node_name
        if u in node2idx and v in node2idx:
            i, j = node2idx[u], node2idx[v]
            adj[0, i, j] = 1.0
            adj[0, j, i] = 1.0

    return adj, node2idx, idx2node


def build_attr_from_pressure_window(pressure_window):
    attr = torch.stack(pressure_window, dim=1)   # [N, W]
    return attr.unsqueeze(0).float()              # [1, N, W]


# ============================================================
#                   OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # 🔹 IPERPARAMETRI
    #hidden_size = trial.suggest_int("hidden_size", 96, 160)
    hidden_size =  132
    #window_size = trial.suggest_int("window_size", 1, 6)
    window_size = 4
    #lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)
    lr = 1e-2
    propag_steps = 7
    topo_dropout = trial.suggest_float("topo_dropout", 0.0, 0.3)

    max_cycle_length = trial.suggest_int("max_cycle_length", 4, 8)

    # ===============================
    # TRAINING
    # ===============================
    env = WNTREnv(INP_PATH, max_steps=MAX_STEPS)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    # --- Topologia (B1/B2) ---
    B1, B2 = compute_B1_B2_for_wn(
        env.wn,
        node2idx,
        max_cycle_length=max_cycle_length,
        device=torch.device(DEVICE)
    )

    model = GGNNCellComplexModel(
        attr_size=window_size,
        hidden_size=hidden_size,
        propag_steps=propag_steps,
        B1=B1,
        B2=B2,
        topo_dropout=topo_dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for ep in range(NUM_EPISODES):
        env.reset()
        sim = env.sim

        pressure_window = []
        train_samples = []

        for step in range(MAX_STEPS):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=AREA, leak_discharge_coefficient=0.75)

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

        for attr, y in train_samples:
            optimizer.zero_grad()
            out = model(attr, adj_matrix).view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    # ===============================
    # TEST (NO RF, ground truth onset)
    # ===============================
    env.reset()
    sim = env.sim

    pressure_window = []
    anomaly_time_series = []

    for step in range(MAX_STEPS):

        if step == env.leak_start_step:
            for ln in env.leak_node_names:
                sim.start_leak(ln, leak_area=AREA, leak_discharge_coefficient=0.75)

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
            u_pred = model(attr_matrix, adj_matrix).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())

    A = np.array(anomaly_time_series)   # [T, N]
    score_per_node = np.sum(np.abs(A), axis=0)

    score = ranking_score_lexicographic(
        score_per_node,
        idx2node,
        env.leak_node_names
    )

    return score


# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="GGNN_CellComplex",
        storage="sqlite:///optuna_ggnn_cellcomplex.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=30)

    print("\n=== OPTUNA FINISHED ===")
    print("Best score:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
