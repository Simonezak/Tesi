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

    num_episodes = 5
    max_steps    = 30
    lr           = 1e-3
    epochs       = 200          # numero di epoch della GGNN
    area = 0.05                 # area dei leak
    WINDOW_SIZE = 4             # quanti snapshot per ogni batch di training
    

    all_snapshots_with_leak = []
    rf_training_data = []         # per RandomForest

    env = WNTREnv(inp_path, max_steps=max_steps)

    # costruisci adiacency matrix e indici UNA VOLTA all'inizio dato che non cambiano
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    # Inizia Simulazione WNTR
    for ep in range(num_episodes):
        print(f"\n--- Episodio {ep+1}/{num_episodes}")
        
        env.reset(with_leak=True)
        wn, sim = env.wn, env.sim

        episode_feature_vectors = []
        pressure_window = [] 

        for step in range(max_steps):

            if step == env.leak_start_step:
                for leak_node in env.leak_node_names:  
                    env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

            sim.step_sim()
            results = sim.get_results()

            # PyG snapshot

            """
            #pressures = data.x[:, 2].cpu().numpy()
            #flows     = data.edge_attr[:, 2].cpu().numpy()
            #vec = np.concatenate([pressures, flows])

            #episode_feature_vectors.append(vec)
            """

            df_pressure = results.node["pressure"]

            current_pressures = torch.tensor(
                [df_pressure.loc[:, name].values[-1] for name in node2idx.keys()],
                dtype=torch.float32
            )

            # aggiorna finestra
            pressure_window.append(current_pressures)

            episode_feature_vectors.append(current_pressures.numpy())

            if len(pressure_window) < WINDOW_SIZE:
                continue

            if len(pressure_window) > WINDOW_SIZE:
                pressure_window.pop(0)

            


            # Label leak per GGNN
            if step < env.leak_start_step:
                continue
            #    y = torch.zeros(data.num_nodes, 1)   # NESSUN NODO in leak
            else:
                df_demand = results.node["demand"]
                df_leak   = results.node.get("leak_demand", None)

                demand = torch.tensor([df_demand.loc[:, name].values[-1] for name in node2idx.keys()],dtype=torch.float32)
                leak = torch.tensor([df_leak.loc[:, name].values[-1] for name in node2idx.keys()],dtype=torch.float32)         
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
            attr_matrix = build_attr_from_pressure_window(pressure_window)

            all_snapshots_with_leak.append({
                "attr":    attr_matrix,
                "adj":     adj_matrix,
                "y":       y
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



    print("\n\n=== TEST PHASE ===")

    test_env = WNTREnv(inp_path, max_steps=max_steps)

    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(test_env.wn)
    test_env.reset(with_leak=True)

    wn, sim = test_env.wn, test_env.sim

    test_snapshots = []
    test_pressure_window = []


    # --------------------
    # 1) LEAK ONSET DETECTION (RandomForest)
    # --------------------

    print("\n--- Leak detection (Random Forest) ---")

    onset_scores = []

    for step in range(max_steps):

        # attiva leak nel momento corretto
        if step == test_env.leak_start_step:
            for leak_node in test_env.leak_node_names:
                test_env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

        sim.step_sim()
        results = sim.get_results()

        df_pressure = results.node["pressure"]

        pressures = np.array([
            df_pressure.loc[:, name].values[-1]
            for name in node2idx.keys()
        ])


        # salvalo in lista
        test_snapshots.append({
            "pressures": pressures
        })

    for snap in test_snapshots:
        data = snap["pressures"]
        prob = rf.predict(data) # da vedere
        onset_scores.append(prob)

    predicted_onset = int(np.argmax(onset_scores))
    print(f"\n Inizio leak stimato allo step: {predicted_onset}")

    # Lista in cui salveremo l’anomalia stimata a ogni step successivo
    anomaly_time_series = []

    # --------------------
    # 2) LEAK LOCALIZATION (GGNN) - PER OGNI STEP DOPO ONSET
    # --------------------

    print("\n--- Leak localization (GGNN) ---")

    for snap in test_snapshots[predicted_onset:]:

        current_pressures = torch.tensor(snap["pressures"], dtype=torch.float32)  # [N]
        test_pressure_window.append(current_pressures)

        if len(test_pressure_window) > WINDOW_SIZE:
            test_pressure_window.pop(0)
        if len(test_pressure_window) < WINDOW_SIZE:
            continue

        attr_matrix = build_attr_from_pressure_window(test_pressure_window)  # [1,N,W]
        with torch.no_grad():
            u_pred = model(attr_matrix, adj_matrix).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())

        """
        # target
        df_demand   = results.node["demand"]
        df_leak     = results.node.get("leak_demand", None)

        demand    = np.array([df_demand.loc[:, name].values[-1]   for name in node2idx.keys()], dtype=np.float32)
        leak      = np.array([df_leak.loc[:, name].values[-1]     for name in node2idx.keys()], dtype=np.float32) if df_leak is not None else np.zeros_like(demand)

        u_target    = (demand + leak).view(-1)

        
        for i in range(len(u_pred)):
            p = float(u_pred[i])
            t = float(u_target[i])
        """

    print("\n\n=== RANKING NODI PER ANOMALIA PERSISTENTE (basato su u_pred) ===")

    A = np.array(anomaly_time_series)
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
        print(f"{idx2node[idx]:<10} {score[idx]:<12.5f} {persist_counts[idx]:<15} {mean_when_active[idx]:<15.5f}")

    print("Nodi leak reali:", test_env.leak_node_names)








if __name__ == "__main__":
    run_GGNN(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Jilin_copy.inp")

