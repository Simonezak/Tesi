import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from wntr_to_pyg import build_pyg_from_wntr, build_static_graph_from_wntr, build_nx_graph_from_wntr, compute_topological_node_features, visualize_snapshot
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, plot_leak_probability_multi, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f, plot_node_demand, plot_edge_flowrate, get_initial_node_demand_limits, get_initial_edge_flow_limits, plot_leak_probability, build_M
from wntr_to_pyg import build_pyg_time_series
from GGNN_Classification import GGNNModel, RandomForestLeakOnsetDetector

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, max_steps=5, hydraulic_timestep=3600, num_leaks=2):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.num_leaks = num_leaks 
        self.sim = None
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.results = None

    def reset(self, with_leak=True):


        # Crea rete
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

 
        # Aggiungi un leak
        self.leak_node_names = []

        if with_leak:
            
            # Prendiamo solo le junctions visto che vogliamo che il leak sia in un nodo
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]
            #self.leak_node_name = np.random.choice(junctions)
            
            num = min(self.num_leaks, len(junctions))
            self.leak_node_names = np.random.choice(junctions, size=num, replace=False).tolist()

            # parametri leak
            self.leak_start_step = np.random.randint(5, 11)

            #self.leak_node_name = "11"
            #self.sim.start_leak(self.leak_node_name, leak_area=area, leak_discharge_coefficient=0.75)
            
            print(f"[LEAK] Nodi selezionati per la perdita: {self.leak_node_names}")
            print(f"[LEAK] Il leak inizierà allo step {self.leak_start_step}")            
    
        else:
            print("[INIT] Nessuna perdita inserita in questo episodio.")


        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        return


def run_GGNN(inp_path):
    """
    Prova del modello GGNN di Leveraging, che
        1) usa solo la pressione dei nodi e matrice di adiacenza per predire leak
        2) non ha topological layer
    """

    num_episodes = 20
    max_steps    = 30
    lr           = 1e-2
    area         = 0.1                 
    epochs       = 300

    all_snapshots_with_leak = []
    rf_training_data = []         # per RandomForest

    env = WNTREnv(inp_path, max_steps=max_steps)

    print("\n=== TRAIN GGNN ===")

    # costruisci adiacency matrix e indici UNA VOLTA all'inizio dato che non cambiano
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    # Inizia Simulazione WNTR
    for ep in range(num_episodes):
        print(f"\n--- Episodio {ep+1}/{num_episodes}")
        
        env.reset(with_leak=True)
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


        y = torch.zeros(len(node2idx), 1)

        for leak_node in env.leak_node_names:
            leak_idx = node2idx[leak_node]
            y[leak_idx] = 1.0


        for t in range(env.leak_start_step, len(episode_feature_vectors)):

            # --- una riga alla volta ---
            p_t = torch.tensor(
                episode_feature_vectors[t],   # [N]
                dtype=torch.float32
            )

            # GGNN input: [1, N, 1]
            attr_matrix = p_t.view(1, -1, 1)

            all_snapshots_with_leak.append({
                "attr": attr_matrix,
                "adj":  adj_matrix,
                "y":    y
            })


        #if ep == 1:
            #sim.plot_results("node", "demand")
            #sim.plot_network_over_time("demand", "flowrate")
            #sim.plot_network()

        rf_training_data.append({
            "feature_vector": episode_feature_vectors,
            "leak_start": env.leak_start_step
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
        attr_size=1,  # solo pressione
        hidden_size=64,
        propag_steps=6
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()


    print("\n=== TRAINING GGNN ===")

    for epoch in range(epochs):
        model.train()

        sample = np.random.choice(all_snapshots_with_leak)

        attr = sample["attr"]
        adj  = sample["adj"]
        y    = sample["y"]  # [N,1]

        # target ora è [1,N]
        target = y.squeeze().float().unsqueeze(0)

        optimizer.zero_grad()
        out = model(attr, adj) # output [1,N]

        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss={loss.item():.4f}")

    # 4️⃣ TEST su nuovo episodio

    print("\n\n=== TEST PHASE ===")

    test_env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(test_env.wn)
    test_env.reset(with_leak=True)
    sim = test_env.sim


    # ============================================================
    # 1) LEAK ONSET DETECTION (Random Forest)
    # ============================================================

    print("\n--- Leak onset detection (Random Forest) ---")

    onset_scores = []

    for step in range(max_steps):

        if step == test_env.leak_start_step:
            for leak_node in test_env.leak_node_names:
                sim.start_leak(
                    leak_node,
                    leak_area=area,
                    leak_discharge_coefficient=0.75
                )

        sim.step_sim()

    results = sim.get_results()
    df_pressure = results.node["pressure"]
    cols = list(node2idx.keys())

    for t in range(len(df_pressure)):
        pressures = df_pressure.loc[:, cols].iloc[t].to_numpy(dtype=np.float32)
        prob = rf.predict(pressures)
        onset_scores.append(prob)

    predicted_onset = int(np.argmax(onset_scores))
    print(f"\n Inizio leak stimato allo step: {predicted_onset}")


    # ============================================================
    # 2) LEAK LOCALIZATION (GGNN) — SNAPSHOT PER SNAPSHOT
    # ============================================================

    print("\n--- Leak localization (GGNN) ---")

    prob_time_series = []

    for t in range(predicted_onset, len(df_pressure)):

        # ---- UNA SOLA RIGA ----
        p_t = df_pressure.loc[:, cols].iloc[t].to_numpy(dtype=np.float32)

        # GGNN input corretto: [1, N, 1]
        attr_matrix = torch.tensor(p_t, dtype=torch.float32).view(1, -1, 1)

        with torch.no_grad():
            logits = model(attr_matrix, adj_matrix).view(-1)   # [N]
            p_pred = torch.sigmoid(logits)                      # [N]

        prob_time_series.append(p_pred.cpu().numpy())


    # ============================================================
    # AGGREGAZIONE TEMPORALE
    # ============================================================
    print("\n=== PROBABILITÀ DELL’ULTIMO SNAPSHOT ===")

    # A = [T', N]
    A = np.stack(prob_time_series, axis=0)

    # prendi SOLO l'ultimo timestep
    last_p = A[-1]     # [N]

    ranking = np.argsort(-last_p)

    print(f"\n{'Nodo':<12} {'p_last':<12}")
    print("-" * 30)

    for i in ranking:
        node_name = idx2node[i]
        print(f"{node_name:<12} {last_p[i]:<12.5f}")

    best_node = ranking[0]
    print(f"\n🔍 Nodo più sospetto (ultimo snapshot): {idx2node[best_node]}")
    print(f"🎯 Nodo reale in leak: {test_env.leak_node_names}")

        
    """
    print("\n=== MEDIA TEMPORALE DELLE PROBABILITÀ ===")

    A = np.stack(prob_time_series, axis=0)   # [T', N]
    mean_p = A.mean(axis=0)                  # [N]

    ranking = np.argsort(-mean_p)

    print(f"\n{'Nodo':<12} {'mean_p':<12}")
    print("-" * 30)

    for i in ranking:
        node_name = idx2node[i]
        print(f"{node_name:<12} {mean_p[i]:<12.5f}")

    best_node = ranking[0]
    print(f"\n🔍 Nodo più sospetto (media probabilità): {idx2node[best_node]}")
    print(f"🎯 Nodo reale in leak: {test_env.leak_node_names}")


    # ============================================================
    # TARGET (solo per debug / valutazione)
    # ============================================================

    y = torch.zeros(len(node2idx))
    for leak_node in test_env.leak_node_names:
        y[node2idx[leak_node]] = 1.0
    """





if __name__ == "__main__":
    run_GGNN(inp_path=r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp")

