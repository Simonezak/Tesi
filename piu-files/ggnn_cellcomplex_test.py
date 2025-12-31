import numpy as np
import torch
import wntr
import networkx as nx
from typing import List, Dict, Tuple

from main_dyn_topologyknown_01 import func_gen_B2_lu
from optuna_GGNN import ranking_score_lexicographic
from evaluation import evaluate_model_across_tests_lexicographic


# -----------------------------
# Simulator import
# -----------------------------
try:
    from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
except Exception:
    from interactive_network_simulator import InteractiveWNTRSimulator


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

        print(f"[LEAK] Nodes: {self.leak_node_names}")
        print(f"[LEAK] Leak start step: {self.leak_start_step}")

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


def build_attr_from_pressure_window(pressure_window: List[torch.Tensor]):
    # pressure_window: list of [N]
    attr = torch.stack(pressure_window, dim=1)  # [N, W]
    return attr.unsqueeze(0).float()            # [1, N, W]


# ============================================================
#                   MODEL IMPORT
# ============================================================

from ggnn_cellcomplex_train import (
    GGNNCellComplexModel,
    compute_B1_B2_for_wn
)


# ============================================================
#                   SINGLE TEST EPISODE
# ============================================================

def run_single_test_episode(
    inp_path: str,
    model: torch.nn.Module,
    max_steps: int,
    window_size: int,
    leak_area: float,
    device: str = "cpu"
):
    env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(device)

    env.reset()
    sim = env.sim

    # ------------------------
    # Run simulation
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
    # GGNN inference (NO RF)
    # ------------------------
    pressure_window = []
    anomaly_time_series = []

    for t in range(env.leak_start_step, len(df_pressure)):
        p = torch.tensor(
            df_pressure.iloc[t][cols].to_numpy(dtype=np.float32),
            device=device
        )
        pressure_window.append(p)

        if len(pressure_window) < window_size:
            continue
        if len(pressure_window) > window_size:
            pressure_window.pop(0)

        attr = build_attr_from_pressure_window(pressure_window).to(device)

        with torch.no_grad():
            u_pred = model(attr, adj_matrix).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())

    A = np.array(anomaly_time_series)       # [T, N]
    score_per_node = np.sum(np.abs(A), axis=0)

    return score_per_node, idx2node, env.leak_node_names


# ============================================================
#                   MULTIPLE TESTS
# ============================================================

def run_multiple_tests(
    inp_path: str,
    model: torch.nn.Module,
    num_test: int = 100,
    max_steps: int = 50,
    window_size: int = 4,
    leak_area: float = 0.1,
    device: str = "cpu",
    X: int = 2
):
    # --- per metriche lessicografiche finali
    scores_per_test = []
    leak_nodes_per_test = []

    # --- score continuo (facoltativo, lo manteniamo)
    localization_scores = []

    for i in range(num_test):
        print(f"\n=== TEST {i+1}/{num_test} ===")

        score_per_node, idx2node, leak_nodes = run_single_test_episode(
            inp_path=inp_path,
            model=model,
            max_steps=max_steps,
            window_size=window_size,
            leak_area=leak_area,
            device=device
        )

        # ---- continuo (come prima)
        loc_score = ranking_score_lexicographic(
            score_per_node,
            idx2node,
            leak_nodes
        )
        localization_scores.append(loc_score)

        # ---- per metriche lessicografiche dense
        scores_per_test.append(score_per_node)
        leak_nodes_per_test.append(leak_nodes)

        print(f"Leak nodes         : {leak_nodes}")
        print(f"Localization score : {loc_score:.4f}")

    localization_scores = np.array(localization_scores)

    # ====================================================
    # METRICHE FINALI LESSICOGRAFICHE (dense rank)
    # ====================================================
    lex_metrics = evaluate_model_across_tests_lexicographic(
        scores_per_test=scores_per_test,
        idx2node=idx2node,
        leak_nodes_per_test=leak_nodes_per_test,
        X=X
    )

    print("\n================ SUMMARY ================")
    print(f"Num test               : {num_test}")

    print("\n--- Localization (continuous score) ---")
    print(f"Mean localization score: {localization_scores.mean():.4f}")
    print(f"Std localization score : {localization_scores.std():.4f}")

    print("\n--- Localization (lexicographic, dense rank) ---")
    for k, v in lex_metrics.items():
        print(f"{k:15s}: {v:.2f}%")

    return localization_scores, lex_metrics



# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = "/home/zagaria/Tesi/Tesi/Networks-found/Jilin_copy_copy.inp"
    model_ckpt = "/home/zagaria/Tesi/Tesi/piu-files/saved_models/ggnn_cellcomplex.pt"

    device = "cpu"

    ckpt = torch.load(model_ckpt, map_location=device)

    # Ricostruisci topologia
    wn = wntr.network.WaterNetworkModel(inp_path)
    _, node2idx, _ = build_static_graph_from_wntr(wn)
    B1, B2 = compute_B1_B2_for_wn(
        wn,
        node2idx,
        max_cycle_length=ckpt["max_cycle_length"],
        device=torch.device(device)
    )

    model = GGNNCellComplexModel(
        attr_size=ckpt["window_size"],
        hidden_size=ckpt["hidden_size"],
        propag_steps=ckpt["propag_steps"],
        B1=B1,
        B2=B2,
        topo_dropout=ckpt.get("topo_dropout", 0.0)
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    run_multiple_tests(
        inp_path=inp_path,
        model=model,
        num_test=100,
        max_steps=50,
        window_size=ckpt["window_size"],
        leak_area=0.1,
        device=device,
        X=2
    )
