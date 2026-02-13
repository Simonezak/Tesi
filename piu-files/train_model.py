import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from wntr_to_pyg import build_pyg_from_wntr, compute_topological_node_features, visualize_snapshot, build_static_graph_from_wntr
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, plot_leak_probability_multi, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f, plot_node_demand, plot_edge_flowrate, get_initial_node_demand_limits, get_initial_edge_flow_limits, plot_leak_probability, build_M
from wntr_to_pyg import build_pyg_time_series
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
    Prova del modello GGNN di Leveraging, che
        1) usa solo la pressione dei nodi e matrice di adiacenza per predire leak
        2) non ha topological layer
    """

    num_episodes = 200
    max_steps    = 50
    lr           = 1e-2
    epochs       = 500
    area = 0.1
    HIDDEN_SIZE = 132
    PROPAG_STEPS = 7
    WINDOW_SIZE = 4 


    all_snapshots_with_leak = []
    rf_training_data = []

    env = WNTREnv(inp_path, max_steps=max_steps)

    # costruisci adiacency matrix e indici UNA VOLTA all'inizio dato che non cambiano
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    # Inizia Simulazione WNTR
    for ep in range(num_episodes):
        print(f"\n--- Episodio {ep+1}/{num_episodes}")
        
        env.reset(num_leaks=2)
        sim = env.sim

        episode_feature_vectors = []

        for step in range(max_steps):

            if step == env.leak_start_step:
                for leak_node in env.leak_node_names:  
                    env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

            sim.step_sim()


            """
            for leak_node in env.leak_node_names:
                leak_idx = node2idx[leak_node]
                leak_val = data.x[leak_idx, 3].item()

                print(f"Step {step}: leak_demand[{leak_node}] = {leak_val:.6f}")
            """       

        results = sim.get_results()

        df_pressure = results.node["pressure"]       # shape [T, N]
        df_demand   = results.node["demand"]         # shape [T, N]
        df_leak     = results.node.get("leak_demand", None)

        # Aggiungi ogni riga del dataframe
        cols = list(node2idx.keys())
        episode_feature_vectors = df_pressure[cols].to_numpy(dtype=np.float32).tolist()

        #if ep == 1:
            #sim.plot_results("node", "demand")
            #sim.plot_network_over_time("demand", "flowrate")
            #sim.plot_network()

        rf_training_data.append({
            "feature_vector": episode_feature_vectors,
            "leak_start": env.leak_start_step
        })


    cols = list(node2idx.keys())

    P = df_pressure[cols].to_numpy(dtype=np.float32)   # [T, N]
    D = df_demand[cols].to_numpy(dtype=np.float32)     # [T, N]

    if df_leak is None:
        L = np.zeros_like(D)
    else:
        L = df_leak[cols].to_numpy(dtype=np.float32)

        
    T, N = P.shape

    for t in range(WINDOW_SIZE - 1, T):

        # finestra pressione [W, N]
        window = P[t - WINDOW_SIZE + 1 : t + 1]     # [W, N]

        # attr_matrix [1, N, W]
        attr_matrix = torch.tensor(
            window.T, dtype=torch.float32
        ).unsqueeze(0)

        # target solo dopo leak onset
        if t < env.leak_start_step:
            continue

        u = D[t] + L[t]                              # [N]
        y = torch.tensor(u, dtype=torch.float32).view(-1, 1)

        all_snapshots_with_leak.append({
            "attr": attr_matrix,
            "adj":  adj_matrix,
            "y":    y
        })





    # ============================================================
    #            TRAIN RANDOM FOREST LEAK-ONSET
    # ============================================================

    print("\n=== TRAINING RANDOM FOREST ===")
    rf = RandomForestLeakOnsetDetector()
    rf.fit(rf_training_data)

    # ============================================================
    #                       TRAIN GGNN
    # ============================================================

    model = GGNNModel(
        attr_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    print("\n=== TRAINING GGNN ===")

    for epoch in range(epochs):

        model.train()

        sample = np.random.choice(all_snapshots_with_leak)

        attr = sample["attr"]
        adj  = sample["adj"]
        y    = sample["y"]  # [N,1]

        # target [1, N]
        target = y.squeeze().float().unsqueeze(0)

        optimizer.zero_grad()
        out = model(attr, adj)   # [1, N]
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss={loss.item():.8f}")


    # ============================================================
    #                 SAVE TRAINED MODELS
    # ============================================================

    import os
    import pickle

    os.makedirs("saved_models", exist_ok=True)

    # ---- salva GGNN ----
    ggnn_path = "saved_models/ggnn_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "attr_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "propag_steps": PROPAG_STEPS
    }, ggnn_path)

    print(f"\n[OK] GGNN salvato in: {ggnn_path}")

    # ---- salva Random Forest ----
    rf_path = "saved_models/rf_leak_onset.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)

    print(f"[OK] RandomForest salvato in: {rf_path}")





if __name__ == "__main__":
    run_GGNN(inp_path=r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp")

