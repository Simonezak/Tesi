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
from GGNN_multi import GGNNModel, RandomForestLeakOnsetDetector

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

def pyg_to_ggnn_inputs(data, pressure_window):
    """
    Converte un PyG Data + finestra temporale delle pressioni
    in input compatibili con il modello GGNN:
    
    - attr_matrix : tensor [1, N, WINDOW_SIZE]
    - adj_matrix  : tensor [1, N, N]
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

def build_incidence_matrix(data):
    """
    Costruisce la matrice B1 del grafo:
    B1[nodi, archi], orientata arbitrariamente.
    """
    N = data.num_nodes
    E = data.edge_index.shape[1]

    B1 = torch.zeros((N, E), dtype=torch.float32)

    src = data.edge_index[0]
    dst = data.edge_index[1]

    for e in range(E):
        s = int(src[e])
        d = int(dst[e])
        B1[s, e] = 1.0
        B1[d, e] = -1.0

    return B1

def localize_leak_udik_nodes(u_vec):
    u = u_vec.view(-1)
    N = len(u)

    residuals = []

    # varianza totale
    total_sq = torch.sum(u*u)

    for i in range(N):
        # residuo = total_sq - u[i]^2
        r = total_sq - u[i]**2
        residuals.append(r.item())

    residuals = np.array(residuals)
    best_node = int(np.argmin(residuals))
    return best_node, residuals




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
    WINDOW_SIZE = 4
    

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
        pressure_window = [] 

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

            current_pressures = data.x[:, 2].cpu()

            # aggiorna finestra
            pressure_window.append(current_pressures)

            if len(pressure_window) < WINDOW_SIZE:
                continue   # aspetta finché non c’è abbastanza storia temporale

            if len(pressure_window) > WINDOW_SIZE:
                pressure_window.pop(0)

            # salva solo se la finestra è completa
            episode_feature_vectors.append(current_pressures.numpy())


            # Label leak per GGNN
            if step < env.leak_start_step:
                continue
            #    y = torch.zeros(data.num_nodes, 1)   # NESSUN NODO in leak
            else:

                demand = data.x[:,1]       # colonna demand
                leak = data.x[:,3]         # colonna leak_demand
                u = demand + leak          # grandezza fisica reale

                y = u.view(-1,1).float()   # shape [N,1]

                """
                y = torch.zeros(data.num_nodes, 1)

                for leak_node in env.leak_node_names:
                    if leak_node in node2idx:
                        y[node2idx[leak_node]] = 1.0       # leak solo dopo l’inizio


                        print(y)
                """

            """
            for leak_node in env.leak_node_names:
                leak_idx = node2idx[leak_node]
                leak_val = data.x[leak_idx, 3].item()

                print(f"Step {step}: leak_demand[{leak_node}] = {leak_val:.6f}")
            """

            # --- PER GGNN: costruisci sample complesso
            attr_matrix, adj_matrix = pyg_to_ggnn_inputs(
                    data, pressure_window=[p for p in pressure_window]
                )


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
        attr_size=WINDOW_SIZE,  # solo pressione
        hidden_size=64,
        propag_steps=6
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


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
    test_pressure_window = []


    # --------------------
    # 1) LEAK ONSET DETECTION (RandomForest)
    # --------------------

    print("\n--- Leak onset detection (Random Forest) ---")

    onset_scores = []

    for step in range(max_steps):

        # attiva leak nel momento corretto
        if step == test_env.leak_start_step:
            area = 0.05
            for leak_node in test_env.leak_node_names:
                test_env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

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
    print(f"\n ⏱️ Inizio leak stimato allo step: {predicted_onset}")

    # Lista in cui salveremo l’anomalia stimata a ogni step successivo
    anomaly_time_series = []


    # --------------------
    # 2) LEAK LOCALIZATION (GGNN) - PER OGNI STEP DOPO ONSET
    # --------------------

    print("\n--- Leak localization (GGNN) ---")

    for snap in test_snapshots[predicted_onset:]:
        
        step = snap["step"]
        data = snap["data"]
        node2idx = snap["node2idx"]
        idx2node = snap["idx2node"]

        current_pressures = data.x[:,2].cpu()
        test_pressure_window.append(current_pressures)

        if len(test_pressure_window) > WINDOW_SIZE:
            test_pressure_window.pop(0)

        if len(test_pressure_window) < WINDOW_SIZE:
            continue

        attr_matrix, adj_matrix = pyg_to_ggnn_inputs(
            data, pressure_window=[p for p in test_pressure_window]
        )


        with torch.no_grad():
            u_pred = model(attr_matrix, adj_matrix).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())

        # target
        demand      = data.x[:,1]
        leak        = data.x[:,3]
        u_target    = (demand + leak).view(-1)

        #cprint(f"\nSTEP {step}")
        #print(f"{'Nodo':<8} {'u_pred':<12} {'u_target':<12} {'diff':<12}")
        for i in range(len(u_pred)):
            p = float(u_pred[i])
            t = float(u_target[i])
            #print(f"{i:<8} {p:<12.5f} {t:<12.5f} {p - t:<12.5f}")

    print("\n\n=== RANKING NODI PER ANOMALIA PERSISTENTE (basato su u_pred) ===")

    # A: [T, N]   T = numero di step dopo onset, N = num_nodi
    A = np.array(anomaly_time_series)   # float64
    T, N = A.shape

    # 1) soglia globale per considerare un nodo "attivo"
    #    ad es. una frazione del max globale
    global_max = A.max()
    threshold = 0.05 * global_max   # puoi regolarla

    # 2) per ogni nodo: quante volte supera la soglia, e quanto è alto in media
    persist_counts = (A > threshold).sum(axis=0)          # [N] numero di step "attivi"
    mean_when_active = np.where(
        persist_counts > 0,
        A * (A > threshold).astype(float),
        0.0
    )
    # media condizionata: somma / count
    sum_when_active = mean_when_active.sum(axis=0)        # [N]
    mean_when_active = np.where(
        persist_counts > 0,
        sum_when_active / persist_counts,
        0.0
    )

    # 3) score combinato: continuità * ampiezza
    #    nodi con leak tendono ad avere molti step attivi e valori non banali
    score = persist_counts * mean_when_active   # [N]

    # 4) ranking decrescente di score
    ranking = np.argsort(-score)

    print(f"\n{'Nodo_idx':<10} {'score':<12} {'count_active':<15} {'mean_active':<15}")
    print("-" * 60)
    for idx in ranking:
        print(f"{idx:<10} {score[idx]:<12.5f} {persist_counts[idx]:<15} {mean_when_active[idx]:<15.5f}")

    print("\nNodo più sospetto (anomalia persistente):", ranking[0])
    print("Nodi leak reali:", test_env.leak_node_names)








if __name__ == "__main__":
    run_GGNN(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Jilin_copy.inp")

