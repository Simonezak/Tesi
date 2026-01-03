import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from wntr_to_pyg import build_pyg_from_wntr, compute_topological_node_features, visualize_snapshot, build_static_graph_from_wntr
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, plot_leak_probability_multi, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f, plot_node_demand, plot_edge_flowrate, get_initial_node_demand_limits, get_initial_edge_flow_limits, plot_leak_probability, build_M
from topological import plot_edge_s_u, plot_edge_Uhat
from GGNN_Regression import GGNNModel, RandomForestLeakOnsetDetector
import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, max_steps=5, hydraulic_timestep=3600):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.sim = None
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.results = None

        

    def reset(self, num_leaks=2):


        # Crea rete
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)


        # Aggiungi un leak
        self.leak_node_names = []
        self.leak_start_step = None
            
        if num_leaks > 0:
            # Prendiamo solo le junctions
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]

            num = min(num_leaks, len(junctions))
            self.leak_node_names = np.random.choice(
                junctions, size=num, replace=False
            ).tolist()

            # Step di inizio leak
            self.leak_start_step = np.random.randint(10, 26)

            print(f"[LEAK] Nodi selezionati per la perdita: {self.leak_node_names}")
            print(f"[LEAK] Il leak inizierà allo step {self.leak_start_step}")
        else:
            print("[INIT] Episodio senza leak")
            
        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        return


def pyg_to_ggnn_inputs(data, pressure_window):
    """
    Converte un PyG Data + finestra temporale delle pressioni
    in input compatibili con il modello GGNN:
    
    - attr_matrix : tensor [1, N, WINDOW_SIZE]
    - adj_matrix  : tensor [1, N, N] Ma l'adiacency matrix non cambia nel tempo vabbe
    """
    
    pressures = torch.stack(pressure_window, dim=1)   # [N, WINDOW_SIZE]
    pressures = pressures.unsqueeze(0).float()        # [1, N, WINDOW_SIZE]

    # ---- Matrice di adiacenza NxN
    N = data.num_nodes
    adj = torch.zeros((N, N), dtype=torch.float32)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0  # grafo non orientato

    adj = adj.view(1, N, N)  # -> [1, N, N]

    return pressures, adj

def build_attr_from_pressure_window(pressure_window):
    """
    Costruisce attr_matrix a partire dalla finestra temporale
    delle pressioni già estratte.

    pressure_window: list di tensor [N]
    Ritorna:
        attr_matrix: [1, N, WINDOW_SIZE]
    """
    # stack temporale: [N, WINDOW_SIZE]
    attr = torch.stack(pressure_window, dim=1)

    # aggiungi dimensione batch
    attr_matrix = attr.unsqueeze(0).float()  # [1, N, WINDOW_SIZE]

    return attr_matrix


def run_GGNN(inp_path):
    """
    GGNN baseline
    - UNICO ciclo sugli episodi
    - stesso training di topo_prova1
    - RF (onset) + GGNN (localizzazione)
    - modello invariato
    - CPU only
    """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


    num_episodes = 50
    max_steps    = 50
    lr           = 1e-2
    area         = 0.1

    HIDDEN_SIZE  = 132
    PROPAG_STEPS = 7
    WINDOW_SIZE  = 1

    # ============================================================
    # ENV + GRAFO STATICO
    # ============================================================

    env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    cols = list(node2idx.keys())

    # ============================================================
    # MODELLI
    # ============================================================

    model = GGNNModel(
        attr_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rf = RandomForestLeakOnsetDetector()
    rf_training_data = []

    print("\n=== TRAINING RF + GGNN (SINGLE EPISODIC LOOP) ===")

    # ============================================================
    # UNICO LOOP EPISODICO
    # ============================================================

    for epoch in range(num_episodes):

        print(f"\n--- Episodio {epoch+1}/{num_episodes}")

        model.train()
        env.reset(num_leaks=2)
        sim = env.sim

        pressure_window = []
        episode_pressures = []

        for step in range(max_steps):

            # attiva leak
            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(
                        ln,
                        leak_area=area,
                        leak_discharge_coefficient=0.75
                    )

            sim.step_sim()
            results = sim.get_results()

            # ------------------------
            # PRESSIONI (per RF)
            # ------------------------
            p_vec = results.node["pressure"].iloc[-1][cols].to_numpy(dtype=np.float32)
            episode_pressures.append(p_vec)

            # ------------------------
            # GGNN (solo dopo onset)
            # ------------------------
            p = torch.tensor(p_vec, dtype=torch.float32)
            pressure_window.append(p)

            if len(pressure_window) > WINDOW_SIZE:
                pressure_window.pop(0)
            if len(pressure_window) < WINDOW_SIZE:
                continue
            if step < env.leak_start_step:
                continue

            attr = build_attr_from_pressure_window(pressure_window)

            demand = results.node["demand"].iloc[-1][cols].values
            leak = results.node.get("leak_demand", None)
            leak = leak.iloc[-1][cols].values if leak is not None else 0
            target = torch.tensor(
                demand + leak,
                dtype=torch.float32
            ).unsqueeze(0)

            optimizer.zero_grad()
            out = model(attr, adj_matrix)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

        # ------------------------
        # AGGIORNA RF (1 sample = 1 episodio)
        # ------------------------
        rf_training_data.append({
            "feature_vector": episode_pressures,
            "leak_start": env.leak_start_step
        })

        # retrain incrementale (semplice ma corretto)
        rf.fit(rf_training_data)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | RF samples: {len(rf_training_data)}")

    # ============================================================
    # SAVE MODELS
    # ============================================================

    import pickle
    os.makedirs("saved_models", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "attr_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "propag_steps": PROPAG_STEPS
    }, "saved_models/ggnn_model_a.pt")

    with open("saved_models/rf_leak_onset_a.pkl", "wb") as f:
        pickle.dump(rf, f)

    print("\n[OK] GGNN e RF salvati")


if __name__ == "__main__":
    run_GGNN(inp_path=r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp")