import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from wntr_to_pyg import build_pyg_from_wntr, build_nx_graph_from_wntr, compute_topological_node_features, visualize_snapshot
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, plot_leak_probability_multi, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f, plot_node_demand, plot_edge_flowrate, get_initial_node_demand_limits, get_initial_edge_flow_limits, plot_leak_probability, build_M
from GNN_LD import GNNLeakDetector, train_model
from GNN_TopoLD import GNNLeakDetectorTopo
from wntr_to_pyg import build_pyg_time_series
from topological import plot_edge_s_u, plot_edge_Uhat
from GGNN_multi_SS import GGNNModel, RandomForestLeakOnsetDetector

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, max_steps=5, hydraulic_timestep=3600, num_leaks=2):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.num_leaks = num_leaks 
        self.sim = None
        self.wn = None
        self.results = None

    def reset(self, with_leak=True):


        # Crea rete
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

 
        # Aggiungi un leak
        self.leak_node_names = []

        if with_leak:
            
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]
            #self.leak_node_name = np.random.choice(junctions)
            
            num = min(self.num_leaks, len(junctions))
            self.leak_node_names = np.random.choice(junctions, size=num, replace=False).tolist()


            # parametri leak
            area = 0.05
            self.leak_start_step = np.random.randint(5, 11)


            #self.leak_node_name = "11"
            print(f"[LEAK] Nodi selezionati per la perdita: {self.leak_node_names}")
            print(f"[LEAK] Il leak inizierà allo step {self.leak_start_step}")

            #self.sim.start_leak(self.leak_node_name, leak_area=area, leak_discharge_coefficient=0.75)
            
        else:
            print("[INIT] Nessuna perdita inserita in questo episodio.")

        # Inizializzazione simulazione WNTR

        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        return


def pyg_to_ggnn_inputs(data):
    """
    Converte un PyG Data generato da build_pyg_from_wntr
    in input compatibili con GGNNModel:
    
    - attr_matrix : tensor [1, N, 1]   (es. pressioni)
    - adj_matrix  : tensor [1, N, N]
    """
    
    # ---- Feature nodali: usa solo la pressione (colonna 2 di data.x)
    pressure = data.x[:, 2].view(1, -1, 1).float()  # shape [1, N, 1]

    # ---- Matrice di adiacenza NxN
    N = data.num_nodes
    adj = torch.zeros((N, N), dtype=torch.float32)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0  # grafo non orientato

    adj = adj.view(1, N, N)  # -> [1, N, N]

    return pressure, adj


def run_GGNN(inp_path):
    """
    Prova del modello GGNN di Leveraging, che
        1) usa solo la pressione dei nodi e matrice di adiacenza per predire leak
        2) non ha topological layer
    """

    num_episodes = 5
    max_steps    = 30
    lr           = 1e-3
    epochs       = 200

    all_snapshots_with_leak = []
    rf_training_data = []         # per RandomForest

    env = WNTREnv(inp_path, max_steps=max_steps)

    print("\n=== TRAIN GGNN ===")

    # Simulazione WNTR
    for ep in range(num_episodes):
        print(f"\n--- Episodio {ep+1}/{num_episodes}")
        
        env.reset(with_leak=True)
        wn, sim = env.wn, env.sim

        episode_feature_vectors = []   # tutti gli step PyG

        for step in range(max_steps):

            if step == env.leak_start_step:
                area = 0.05
                for leak_node in env.leak_node_names:  
                    env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

            sim.step_sim()
            results = sim.get_results()

            # PyG snapshot
            data, node2idx, idx2node, _, _ = build_pyg_from_wntr(wn, results)

            # --- PER RF: aggiungi SOLO data (PyG Data)
            #pressures = data.x[:, 2].cpu().numpy()
            #flows     = data.edge_attr[:, 2].cpu().numpy()
            #vec = np.concatenate([pressures, flows])

            #episode_feature_vectors.append(vec)

            pressures = data.x[:, 2].cpu().numpy()
            episode_feature_vectors.append(pressures)


            # Label leak per GGNN
            if step < env.leak_start_step:
                continue
            #    y = torch.zeros(data.num_nodes, 1)   # NESSUN NODO in leak
            else:
                y = torch.zeros(data.num_nodes, 1)

                for leak_node in env.leak_node_names:
                    leak_idx = node2idx[leak_node]
                    y[leak_idx] = 1.0
                """
                for leak_node in env.leak_node_names:
                    if leak_node in node2idx:
                        y[node2idx[leak_node]] = 1.0       # leak solo dopo l’inizio
                        leak = data.x[:,3]         # colonna leak_demand
                        u = leak          # grandezza fisica reale

                        y = u.view(-1,1).float()   # shape [N,1]

                        #print(y)
                """

            """
            for leak_node in env.leak_node_names:
                leak_idx = node2idx[leak_node]
                leak_val = data.x[leak_idx, 3].item()

                print(f"Step {step}: leak_demand[{leak_node}] = {leak_val:.6f}")
            """

            # --- PER GGNN: costruisci sample complesso
            attr_matrix, adj_matrix = pyg_to_ggnn_inputs(data)

            all_snapshots_with_leak.append({
                "attr":    attr_matrix,
                "adj":     adj_matrix,
                "y":       y,
                "data":    data,
                "node2idx": node2idx,
                "idx2node": idx2node
            })


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
        total_loss = 0
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
    test_env.reset(with_leak=True)
    wn, sim = test_env.wn, test_env.sim

    test_snapshots = []   # 🔥 LISTA DI TUTTI GLI SNAPSHOT

    # --------------------
    # 1) LEAK ONSET DETECTION (RandomForest)
    # --------------------

    print("\n--- Leak onset detection (Random Forest) ---")

    onset_scores = []

    for step in range(max_steps):

        # attiva leak nel momento corretto
        if step == test_env.leak_start_step:
            area = 0.09
            for leak_node in test_env.leak_node_names:
                test_env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.85)

        sim.step_sim()
        results = sim.get_results()

        # conversione a PyG
        data, node2idx, idx2node, _, _ = build_pyg_from_wntr(wn, results)

        # salvalo in lista
        test_snapshots.append({
            "step": step,
            "data": data,
            "node2idx": node2idx,
            "idx2node": idx2node,
            "results": results,
        })

    for snap in test_snapshots:
        data = snap["data"]
        prob = rf.predict(data)
        onset_scores.append(prob)

    predicted_onset = int(np.argmax(onset_scores))
    print(f"\n Inizio leak stimato allo step: {predicted_onset}")

    # Lista in cui salveremo l’anomalia stimata a ogni step successivo
    anomaly_time_series = []


    # --------------------
    # 2) LEAK LOCALIZATION (GGNN) - PER OGNI STEP DOPO ONSET
    # --------------------

    print("\n--- Leak localization (GGNN) ---")

    anomaly_time_series = []
    prob_time_series = []

    for snap in test_snapshots[predicted_onset:]:
        
        step = snap["step"]
        data = snap["data"]
        node2idx = snap["node2idx"]
        idx2node = snap["idx2node"]

        attr_matrix, adj_matrix = pyg_to_ggnn_inputs(data)

        """
        with torch.no_grad():
            u_pred = model(attr_matrix, adj_matrix).view(-1)

            TH = torch.quantile(u_pred, 0.8) * 0.5   # metà dell'80th percentile
            u_pred = torch.where(u_pred > TH, u_pred, torch.zeros_like(u_pred))
        """
        with torch.no_grad():
            logits = model(attr_matrix, adj_matrix).view(-1)     # shape [N]
            p_pred = torch.sigmoid(logits)                       # probabilità

        #anomaly_time_series.append(u_pred.cpu().numpy())

                # salva per media temporale
        prob_time_series.append(p_pred.cpu().numpy())

        # ----- target binario -----
        y = torch.zeros(len(p_pred))
        for leak_node in test_env.leak_node_names:
            leak_idx = node2idx[leak_node]
            y[leak_idx] = 1.0

        print(f"\nSTEP {step}")
        print(f"{'Nodo':<12} {'p_pred':<12} {'target':<12}")
        print("-"*35)
        
        for i in range(len(p_pred)):
            node_name = idx2node[i]
            pp = float(p_pred[i])
            tt = float(y[i])
            print(f"{node_name:<12} {pp:<12.5f} {tt:<12.5f}")

    # -----------------------------
    #   MEDIA DELLE PROBABILITÀ
    # -----------------------------

    print("\n\n=== MEDIA TEMPORALE DELLE PROBABILITÀ ===")

    # array [T, N]
    A = np.array(prob_time_series)

    # media (non somma, perché classificazione)
    mean_p = A.mean(axis=0)

    # ranking nodi più sospetti
    ranking = np.argsort(-mean_p)

    print(f"\n{'Nodo':<12} {'mean_p':<12}")
    print("-"*30)

    for i in ranking:
        node_name = idx2node[i]
        print(f"{node_name:<12} {mean_p[i]:<12.5f}")

    best_node = ranking[0]
    print(f"\n🔍 Nodo più sospetto (media probabilità): {idx2node[best_node]}")
    print(f"🎯 Nodo reale in leak: {test_env.leak_node_names}")


    """
    # target
    leak        = data.x[:,3]
    u_target    = (leak).view(-1)

    print(f"\nSTEP {step}")
    print(f"{'Nodo':<8} {'u_pred':<12} {'u_target':<12} {'diff':<12}")
    for i in range(len(u_pred)):
        node_name = idx2node[i]               # nome reale del nodo
        p = float(u_pred[i])
        t = float(u_target[i])
        print(f"{node_name:<12} {p:<12.5f} {t:<12.5f} {p - t:<12.5f}")
    
    # Converti lista → array [T, N]
    A = np.array(anomaly_time_series)   # shape: (num_steps_after_onset, num_nodes)

    # Somma temporale delle anomalie
    s_u = A.sum(axis=0)                 # shape: (num_nodes,)

    # Ordina i nodi dal più sospetto al meno
    ranking = np.argsort(-s_u)

    print("\n\n=== TOP 10 nodi più sospetti (con nome reale) ===")

    top_k = 10
    top_nodes = ranking[:top_k]

    print(f"\n{'Pos':<5} {'Nodo':<15} {'Anomalia cumulata':<20}")
    print("-"*55)

    for pos, idx in enumerate(top_nodes, start=1):
        node_name = idx2node[idx]      # 🔥 converti indice → nome reale WK
        print(f"{pos:<5} {node_name:<15} {s_u[idx]:<20.5f}")

    print(f"\n Nodi reali in leak: {test_env.leak_node_names}")
    """









if __name__ == "__main__":
    run_GGNN(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Jilin_copy.inp")

