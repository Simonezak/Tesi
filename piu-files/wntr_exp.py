import torch
import torch.optim as optim
import torch.nn as nn
import random
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import numpy as np
import networkx as nx


from data_utils.wntr_to_pyg import build_pyg_from_wntr, build_nx_graph_from_wntr, compute_topological_node_features, visualize_snapshot
from actions import open_pipe, close_pipe, close_all_pipes, noop
import matplotlib.pyplot as plt
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f, plot_node_demand, plot_edge_flowrate, get_initial_node_demand_limits, get_initial_edge_flow_limits, plot_leak_probability
from GNN_LD import GNNLeakDetector, train_model
from GNN_TopoLD import GNNLeakDetectorTopo

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, max_steps=5, hydraulic_timestep=3600):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.sim = None
        self.wn = None
        self.results = None
        self.done = False

    def reset(self, with_leak=True):
        import wntr

        # ===============================
        # Crea rete base
        # ===============================
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

        # ===============================
        # Aggiungi un leak realistico
        # ===============================

        self.leak_node_name = None
        if with_leak:
            
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]
            self.leak_node_name = np.random.choice(junctions)
            
            self.leak_node_name = "11"
            print(f"[LEAK] Nodo selezionato per la perdita: {self.leak_node_name}")

            # parametri leak
            area = 0.05
            self.sim.start_leak(self.leak_node_name, leak_area=area, leak_discharge_coefficient=0.75)
            
            print(f"[LEAK] Aggiunta perdita con area={area:.3e} m²")
        else:
            print("[INIT] Nessuna perdita inserita in questo episodio.")

        # ===============================
        # 3️⃣ Inizializza simulatore WNTR
        # ===============================
        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        return



def run_wntr_experiment(inp_path):

    # ---------------------------
    # 1️⃣ Setup ambiente
    # ---------------------------
    print("\n=== 💧 Simulazione WNTR (Dynamic Topology + Leak) ===")
    env = WNTREnv(inp_path, max_steps=5, hydraulic_timestep=3600)
    env.reset(with_leak=True)
    wn = env.wn
    sim = env.sim

    # Primo Step

    step = 1
    print(f" Step {step}/{env.max_steps}")
    sim.step_sim()
    
    results = sim.get_results()

    G, coords = build_nx_graph_from_wntr(wn, results)
    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)

    # Calcolo matrice dei flussi
    f = construct_matrix_f(wn, results)

    # Calcolo flusso per poligono e limiti iniziali per confrontare
    f_polygons = compute_polygon_flux(f, B2, False)
    f_polygons_abs = compute_polygon_flux(f, B2, True)

    vmin, vmax = get_inital_polygons_flux_limits(f_polygons)
    vmin2, vmax2 = get_initial_node_demand_limits(G)
    vmin3, vmax3 = get_initial_edge_flow_limits(f)

    # Ottengo Coordinate e leak node per la funzione
    leak_node = wn.get_node(env.leak_node_name)
    
    # Plot Demand nodi e Flowrate archi
    plot_node_demand(G, coords, vmin2, vmax2, step=step)
    # è normale che in plot edge flowrate gli archi abbiano segno diverso rispetto a come sono in plot cell complex flux, perchè
    # la direzione viene poi cambiata qunado viene calcolato f_polygons, perche B2 da un verso arbitrario all'edge in base alla direzione che vuole dare al poligono
    plot_edge_flowrate(G, coords, f, vmin3, vmax3, step=step)

    # Plot poligoni prima senza valori assoluti e poi con valori assoluti
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin, vmax, leak_node, step)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons_abs, vmin, vmax, leak_node, step)


    # ---------------------------
    # 2️⃣ Esegui alcuni step di simulazione
    # ---------------------------
    for step in range(2, env.max_steps + 1):
        print(f" Step {step}/{env.max_steps}")
        #if step == 4:
        #    leak_node_name = "11"
        #    print(f"[LEAK] Nodo selezionato per la perdita: {leak_node_name}")

            # parametri leak
        #    area = 0.05
        #    sim.start_leak(leak_node_name, leak_area=area, leak_discharge_coefficient=0.75)

        sim.step_sim()
        results = sim.get_results()

    # ---------------------------
    # 3️⃣ Ricalcolo metriche diverse per l'ultimo step
    # ---------------------------

    G, coords = build_nx_graph_from_wntr(wn, results)

    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=7)
    #filtered_cycles = [c for c in selected_cycles if len(c) <= 5]

    f = construct_matrix_f(wn, results)

    f_polygons = compute_polygon_flux(f, B2, False)
    f_polygons_abs = compute_polygon_flux(f, B2, True)
    
    # Visualizza con la nuova funzione
    plot_node_demand(G, coords, vmin2, vmax2, step=step)
    plot_edge_flowrate(G, coords, f, vmin3, vmax3, step=step)

    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin, vmax, leak_node, step)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons_abs, vmin, vmax, leak_node, step)



def run_GNN_topo_comparison(inp_path):
    """
    - genera i grafi PyG agli step 1..5
    - allena entrambi i modelli su tutti gli step
    - valuta e plotta a step 1 e step 5:
        * plot_leak_probability (entrambi)
        * plot_node_demand / plot_edge_flowrate / plot_cell_complex_flux (contesto)
    """

    # ---------------------------
    # Parametri interni default
    # ---------------------------
    max_steps   = 5
    epochs      = 50
    lr          = 1e-3
    hidden_dim  = 64
    topo_proj   = 32
    dropout     = 0.2

    # ---------------------------
    # 0) Setup ambiente + leak
    # ---------------------------
    print("\n=== 💧 Confronto GCN semplice vs GCN+TopoLayer ===")
    env = WNTREnv(inp_path, max_steps=max_steps, hydraulic_timestep=3600)
    env.reset(with_leak=True)
    wn, sim = env.wn, env.sim
    leak_name = env.leak_node_name
    print(f"[INFO] Leak at node: {leak_name}")

    # ---------------------------
    # 1) Raccolta dataset (1..max_steps)
    # ---------------------------
    graphs = []      # Data PyG
    labels = []      # y per-nodo
    aux    = []      # (G, coords, results, B1,B2,f,f_polygons, node2idx, idx2node)

    for step in range(1, max_steps + 1):
        print(f"  > Sim step {step}/{max_steps}")
        sim.step_sim()
        results = sim.get_results()

        data, node2idx, idx2node, edge2idx, idx2edge = build_pyg_from_wntr(wn, results)

        # Label per-nodo
        y = torch.zeros(data.num_nodes, 1, dtype=torch.float32)
        if leak_name in node2idx:
            y[node2idx[leak_name]] = 1.0
        else:
            print(f"[WARN] Leak node {leak_name} non presente nel grafo PyG allo step {step}")

        graphs.append(data)
        labels.append(y)

        # Oggetti per i plot
        G, coords = build_nx_graph_from_wntr(wn, results)
        B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)
        f = construct_matrix_f(wn, results)
        f_polygons = compute_polygon_flux(f, B2, False)

        aux.append((G, coords, results, B1, B2, f, f_polygons, node2idx, idx2node))

    # ---------------------------
    # 2) Modelli
    # ---------------------------
    sample = graphs[0]
    node_in_dim = sample.x.shape[1]
    topo_in_dim = getattr(sample, "topo", None).shape[1] if hasattr(sample, "topo") and sample.topo is not None else 0

    model_plain = GNNLeakDetector(node_in_dim=node_in_dim, hidden_dim=hidden_dim, dropout=dropout)
    model_topo  = GNNLeakDetectorTopo(node_in_dim=node_in_dim, topo_in_dim=topo_in_dim,
                                      hidden_dim=hidden_dim, topo_proj_dim=topo_proj, dropout=dropout)

    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=lr)
    opt_topo  = torch.optim.Adam(model_topo.parameters(),  lr=lr)
    loss_fn = nn.BCELoss()

    # ---------------------------
    # 3) Training (stessa procedura per entrambi)
    # ---------------------------

    print("\n[TRAIN] GCN semplice")
    train_model(model_plain, opt_plain, graphs, labels, epochs=epochs, name="GCN")

    print("\n[TRAIN] GCN + TopoLayer")
    train_model(model_topo,  opt_topo,  graphs, labels, epochs=epochs, name="GCN+Topo")

    # ---------------------------
    # 4) Valutazione & Plot a step 1 e 5
    # ---------------------------

    return model_plain, model_topo, graphs


def run_GNN_topo_comparison_multi(inp_path):
    """
    Esegue training GNN su più episodi (ogni episodio con leak diverso),
    raccogliendo tutti gli snapshot di tutti gli step come dataset.
    """

    num_episodes=3
    max_steps=5
    lr=1e-3
    hidden_dim=64
    epochs=50

    # Questa lista conterrà tutti gli snapshot di tutti gli episodi
    all_snapshots = []

    # Istanziamento Ambiente
    env = WNTREnv(inp_path, max_steps=max_steps, hydraulic_timestep=3600)
    

    print(f"Inizializzato GNN. \nNumero Episodi: {num_episodes} \nNumero Step: {max_steps} \n\n")

    # Loop Episodi
    for ep in range(num_episodes):
        print(f"Episodio {ep+1}/{num_episodes}")

        # Reset ambiente e Leak
        env.reset(with_leak=True)
        wn, sim = env.wn, env.sim
        leak_node = env.leak_node_name
        print(f"Leak al nodo: {leak_node}")

        # Loop degli step della simulazione
        for step in range(max_steps):

            sim.step_sim()
            results = sim.get_results()

            data, node2idx, idx2node, edge2idx, idx2edge = build_pyg_from_wntr(wn, results)

            # --- Calcolo feature topologiche
            topo_feats, node_order, B1, B2 = compute_topological_node_features(wn, results)

            # Aggiungi le feature topologiche al grafo
            

            # Label: 1 solo sul nodo del leak
            y = torch.zeros(data.num_nodes, 1)
            if leak_node in node2idx:
                y[node2idx[leak_node]] = 1.0
            else:
                print(f"Step {step}, Leak node {leak_node} non presente nel grafo")
            
            data.y = y
            data.episode_id = ep
            data.step = step
            data.topo = torch.tensor(topo_feats, dtype=torch.float32)

            """
            Ogni snapshot quindi contiene: 
            - x
            - edge_index
            - edge_attr
            - num_nodes
            - num_edges
            - y
            - episode_id
            - step
            - pipe_edge_idx
            - pipe_open_mask
            - pipe_names
            - data.num_pipes

            Ci sono un sacco di variabili ma l'importante è che vengano prese solo quelle
            giuste dal modello per il training e non cose come l'episode_id altrimenti overfitta
            """

            #print(f"[EP{ep} STEP{step}] data keys: {list(data.keys())}")

            all_snapshots.append(data)

    visualize_snapshot(all_snapshots, episode_id=2, step=1, wn=wn, results=results)

    # Setup modelli
    node_in_dim = all_snapshots[0].x.shape[1]
    edge_in_dim = all_snapshots[0].edge_attr.shape[1]
    
    # GNN semplice
    model_plain = GNNLeakDetector(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=lr)

    # GNN + Topological
    topo_in_dim = all_snapshots[0].topo.shape[1]  # 2: degree + node_cycle_flux

    model_topo  = GNNLeakDetectorTopo(node_in_dim=node_in_dim, topo_in_dim=topo_in_dim, edge_in_dim=edge_in_dim)    
    opt_topo  = torch.optim.Adam(model_topo.parameters(),  lr=lr)


    print("\n[TRAIN] GCN semplice")
    train_model(model_plain, opt_plain, all_snapshots, epochs=epochs, name="GCN")

    print("\n[TRAIN] GCN + TopoLayer")
    train_model(model_topo,  opt_topo,  all_snapshots, epochs=epochs, name="GCN+Topo")

    #
    # TESTING
    #

    print("\n --- Prediction su nuovo episodio ---")

    test_env = WNTREnv(inp_path, max_steps=max_steps)
    test_env.reset(with_leak=True)
    wn, sim = test_env.wn, test_env.sim
    leak_node = wn.get_node(test_env.leak_node_name)

    for step in range(max_steps):
        sim.step_sim()
        results = sim.get_results()

    # Metriche da calcolare per i plot
    G, coords = build_nx_graph_from_wntr(wn, results)
    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)
    f = construct_matrix_f(wn, results)
    f_polygons = compute_polygon_flux(f, B2, abs=False)
    f_polygons_abs = compute_polygon_flux(f, B2, abs=True)

    vmin_p, vmax_p = get_inital_polygons_flux_limits(f_polygons)
    vmin_n, vmax_n = get_initial_node_demand_limits(G)
    vmin_e, vmax_e = get_initial_edge_flow_limits(f)

    plot_node_demand(G, coords, vmin_n, vmax_n, test=True)
    plot_edge_flowrate(G, coords, f, vmin_e, vmax_e, test=True)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin_p, vmax_p, leak_node, test=True)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons_abs, vmin_p, vmax_p, leak_node, test=True)


    data, node2idx, idx2node, edge2idx, idx2edge = build_pyg_from_wntr(wn, results)

    # ==========================================
    # 🔹 EVALUATE: Modello Plain
    # ==========================================
    print("\nValutazione modello GCN semplice")
    model_plain.eval()
    with torch.no_grad():
        preds_plain = model_plain(data)

    probs_plain = preds_plain.squeeze().detach().cpu()
    topk_plain = torch.topk(probs_plain, k=3)

    print(f"Leak reale: {leak_node}")
    for rank, (idx, val) in enumerate(zip(topk_plain.indices.tolist(), topk_plain.values.tolist()), start=1):
        node_name = idx2node[idx]
        print(f" {rank}. Nodo {node_name} → prob = {val:.4f}")

    plot_leak_probability(G, coords, preds_plain, node2idx[leak_node.name])

    # ==========================================
    # 🔹 EVALUATE: Modello Topologico
    # ==========================================
    print("\nValutazione modello GNN + Topological ")
    topo_feats, node_order, B1, B2 = compute_topological_node_features(wn, results)
    data.topo = torch.tensor(topo_feats, dtype=torch.float32)
    model_topo.eval()
    with torch.no_grad():
        preds_topo = model_topo(data)

    probs_topo = preds_topo.squeeze().detach().cpu()
    topk_topo = torch.topk(probs_topo, k=3)

    print(f"Leak reale: {leak_node}")
    for rank, (idx, val) in enumerate(zip(topk_topo.indices.tolist(), topk_topo.values.tolist()), start=1):
        node_name = idx2node[idx]
        print(f" {rank}. Nodo {node_name} → prob = {val:.4f}")

    plot_leak_probability(G, coords, preds_topo, node2idx[leak_node.name])


if __name__ == "__main__":
    run_GNN_topo_comparison_multi(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Jilin.inp")

