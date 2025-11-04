import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from data_utils.wntr_to_pyg import build_pyg_from_wntr, build_nx_graph_from_wntr, compute_topological_node_features, visualize_snapshot
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

    def reset(self, with_leak=True):


        # Crea rete
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

 
        # Aggiungi un leak
        self.leak_node_name = None
        if with_leak:
            
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]
            self.leak_node_name = np.random.choice(junctions)
            
            #self.leak_node_name = "11"
            print(f"[LEAK] Nodo selezionato per la perdita: {self.leak_node_name}")

            # parametri leak
            area = 0.05
            self.sim.start_leak(self.leak_node_name, leak_area=area, leak_discharge_coefficient=0.75)
            
            print(f"[LEAK] Aggiunta perdita con area={area:.3e} m²")
        else:
            print("[INIT] Nessuna perdita inserita in questo episodio.")

        # Inizializzazione simulazione WNTR

        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        return



def run_wntr_experiment(inp_path):
    """
    Funzione Main, solo plot del Water Network per vedere lo stato
    """

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
    # Ricalcolo metriche diverse per l'ultimo step
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



def run_GNN_topo_comparison_multi(inp_path):
    """
    Funzione Main
    contiene sia modello base che topo. Confronta entrambi i modelli trainando 
    su piu episodi e raccogliendo tutti gli snapshot di tutti gli step come dataset.
    Infine predice il nodo con il leak per provare l'accuracy dei due modelli

    Ogni snapshot contiene: 
        - x
        - y
        - edge_index
        - edge_attr
        - episode_id
        - step

    le variabili che sono usate per il training per ora sono solo x,y e edge_index (quest'ultimo per formare i link)

    episode_id e step sono usati per il debug, con la funzione visualize_snapshot si possono visuallizare per quel passo specifico
    tutti i grafici che mi hanno richiesto

    il tempo funziona correttamente e viene azzerato correttamente ad ogni reset()

    alla fine viene fatta un evaluation e una prediction e vengono prima mostrati i grafici richiesti e poi uno per mostrare
    graficamente le probabilità calcolate con il predict

    """

    num_episodes=3
    max_steps=8
    lr=1e-3
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

            Ci sono un sacco di variabili ma l'importante è che vengano prese solo quelle
            giuste dal modello per il training e non cose come l'episode_id altrimenti overfitta
            """

            #print(f"[EP{ep} STEP{step}] data keys: {list(data.keys())}")

            all_snapshots.append(data)
        
        #sim.plot_results("node", "demand")
        sim.plot_network_over_time("demand", "flowrate")



    visualize_snapshot(all_snapshots, episode_id=2, step=1, wn=wn, results=results)

    # Setup modelli
    node_in_dim = all_snapshots[0].x.shape[1]
    
    # GNN semplice
    model_plain = GNNLeakDetector(node_in_dim=node_in_dim)
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=lr)

    # GNN + Topological
    topo_in_dim = all_snapshots[0].topo.shape[1]  # 2: degree + node_cycle_flux

    model_topo  = GNNLeakDetectorTopo(node_in_dim=node_in_dim, topo_in_dim=topo_in_dim)    
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
    # EVALUATE: Modello Plain
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
    # EVALUATE: Modello Topologico
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
    run_GNN_topo_comparison_multi(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Jilin_copy.inp")

