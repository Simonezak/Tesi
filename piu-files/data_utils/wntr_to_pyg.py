import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import wntr
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from wntr.network.elements import LinkStatus, Junction
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f, plot_node_demand, plot_edge_flowrate, get_initial_node_demand_limits, get_initial_edge_flow_limits

import networkx as nx


def build_nx_graph_from_wntr(wn, results=None, timestep_index=-1):
    """
    Costruisce un grafo NetworkX con i nomi originali dei nodi WNTR.
    Compatibile con func_gen_B2_lu (usa mappa interna).
    
    Args:
        wn : wntr.network.WaterNetworkModel
        results : wntr.sim.results.SimulationResults | None
        timestep_index : int, default=-1

    Returns:
        G : networkx.Graph
            Grafo con i nodi come nomi originali (es. 'J1', 'R1', 'T1').
        coords : np.ndarray
            Array (n_nodes, 2) con le coordinate (x, y) di ciascun nodo.
    """

    # ===============================
    # 1️⃣ Inizializza grafo base
    # ===============================
    G = nx.Graph()

    # ---- Lettura dati dai risultati (se disponibili) ----
    df_demand = getattr(results.node, "get", lambda *_: None)("demand", None) if results else None
    df_pressure = getattr(results.node, "get", lambda *_: None)("pressure", None) if results else None
    df_leak = getattr(results.node, "get", lambda *_: None)("leak_demand", None) if results else None
    df_flow = getattr(results.link, "get", lambda *_: None)("flowrate", None) if results else None

    # ===============================
    # 2️⃣ Nodi
    # ===============================
    for node_name, node in wn.nodes():
        elev = float(getattr(node, "elevation", 0.0))
        demand = float(df_demand.iloc[timestep_index][node_name]) if df_demand is not None else 0.0
        pressure = float(df_pressure.iloc[timestep_index][node_name]) if df_pressure is not None else 0.0
        leak_dem = float(df_leak.iloc[timestep_index][node_name]) if df_leak is not None else 0.0

        G.add_node(
            node_name,
            pos=node.coordinates if hasattr(node, "coordinates") else (0.0, 0.0),
            elevation=elev,
            demand=demand,
            pressure=pressure,
            leak_demand=leak_dem,
            type=node.__class__.__name__,
        )

    # ===============================
    # 3️⃣ Archi
    # ===============================
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start, end = pipe.start_node_name, pipe.end_node_name
        if start not in G or end not in G:
            continue

        length = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow = float(df_flow.iloc[timestep_index][pipe_name]) if df_flow is not None else 0.0

        G.add_edge(
            start,
            end,
            pipe_name=pipe_name,
            length=length,
            diameter=diameter,
            flowrate=flow
        )

    # ===============================
    # 4️⃣ Coordinate e compatibilità con func_gen_B2_lu
    # ===============================

    # mappatura interna per i cicli topologici
    mapping = {name: i for i, name in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping, copy=True)

    # coordinate in ordine numerico coerente con la mappa
    coords = np.array([
        wn.get_node(orig_name).coordinates if hasattr(wn.get_node(orig_name), "coordinates") else (0.0, 0.0)
        for orig_name in wn.node_name_list if orig_name in mapping
    ])

    return G, coords





@dataclass
class GraphFeatureConfig:
    node_features: Tuple[str, ...] = ("elevation", "demand", "pressure", "leak_demand")
    edge_features: Tuple[str, ...] = ("length", "diameter", "flowrate")
    undirected: bool = True


def safe_get(df, col, default=0.0):
    """Legge in modo sicuro un valore da un DataFrame dei risultati (anche se parziale)."""
    if df is None or col not in df.columns or len(df) == 0:
        return default
    # Se non specificato, prende l’ultimo timestep disponibile
    try:
        return float(df.iloc[-1][col])
    except Exception:
        return default


def build_pyg_from_wntr(wn, results, cfg: GraphFeatureConfig = GraphFeatureConfig()):
    # ---- nodi ----
    node_names: List[str] = [name for name, _ in wn.nodes()]
    node2idx = {name: i for i, name in enumerate(node_names)}
    idx2node = {i: name for name, i in node2idx.items()}

    # Lettura attributi di ogni nodo
    elev, demand, pressure, leak_dem = [], [], [], []
    df_demand: Optional[pd.DataFrame] = results.node.get("demand", None)
    df_pressure: Optional[pd.DataFrame] = results.node.get("pressure", None)
    df_leak: Optional[pd.DataFrame] = results.node.get("leak_demand", None)

    for name in node_names:
        n = wn.get_node(name)
        elev.append(float(getattr(n, "elevation", 0.0)))
        demand.append(safe_get(df_demand, name))
        pressure.append(safe_get(df_pressure, name))
        leak_dem.append(safe_get(df_leak, name))
        x = np.stack([elev, demand, pressure, leak_dem], axis=1)
        x_torch = torch.tensor(x, dtype=torch.float32)

    # ---- archi (pipes) ----
    edge_index_list: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []

    df_flow = results.link.get("flowrate", None)

    lengths, diameters, flows, starts, ends, statuses = [], [], [], [], [], []


    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        u_name, v_name = pipe.start_node_name, pipe.end_node_name
        if u_name not in node2idx or v_name not in node2idx:
            continue

        u, v = node2idx[u_name], node2idx[v_name]
        length = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow = safe_get(df_flow, pipe_name)

        # aggiungi forward edge
        edge_index_list.append((u, v))
        edge_attrs.append([length, diameter, flow])

        lengths.append(length)
        diameters.append(diameter)
        flows.append(flow)
        starts.append(u_name)
        ends.append(v_name)

        # se grafo non orientato, aggiungi anche reverse
        if cfg.undirected:
            edge_index_list.append((v, u))
            edge_attrs.append([length, diameter, flow])

     
    # ---- costruisci Data PyG ----
    edge_index = torch.tensor(np.array(edge_index_list, dtype=np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float32)

    data = Data(x=x_torch, edge_index=edge_index, edge_attr=edge_attr)

    edge_names = wn.pipe_name_list
    edge2idx = {name: i for i, name in enumerate(edge_names)}
    idx2edge = {i: name for name, i in edge2idx.items()}

    return data, node2idx, idx2node, edge2idx, idx2edge


def compute_topological_node_features(wn, results, max_cycle_length: int = 8, abs_flux: bool = False):
    """
    Calcola feature topologiche per nodo basate su B1, B2 e flussi poligonali.
    """
    # 1️⃣ Costruisci grafo da WNTR
    G, coords = build_nx_graph_from_wntr(wn, results)

    # 2️⃣ Calcola matrici topologiche (boundary operators)
    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=max_cycle_length)

    # 3️⃣ Costruisci matrice di flussi f sugli archi
    f = construct_matrix_f(wn, results)

    # 4️⃣ Calcola flussi sui poligoni (celle 2D)
    f_polygons = compute_polygon_flux(f, B2, abs=abs_flux)

    # 5️⃣ Propagazione cicli→archi→nodi
    edge_from_poly = np.abs(B2) @ np.abs(f_polygons)
    if B1.shape[0] == edge_from_poly.shape[0]:
        node_cycle_flux = (np.abs(B1).T @ edge_from_poly).reshape(-1)
    elif B1.shape[1] == edge_from_poly.shape[0]:
        node_cycle_flux = (np.abs(B1) @ edge_from_poly).reshape(-1)
    else:
        raise ValueError(f"[TopoFeatures] B1 shape {B1.shape} incompatible with edges {edge_from_poly.shape[0]}")

    # 6️⃣ Grado dei nodi (local centrality)
    deg = np.array([G.degree(n) for n in G.nodes()], dtype=float)

    # 7️⃣ Stack finale [N,2]
    topo_feats = np.stack([deg, node_cycle_flux], axis=1)
    node_order = list(G.nodes())

    return topo_feats, node_order, B1, B2




def visualize_snapshot(all_snapshots, episode_id, step, wn, results):
    """
    Visualizza lo stato WN corrispondente a uno snapshot specifico
    """

    # Trova lo snapshot corrispondente
    snap = next(
        (d for d in all_snapshots
         if getattr(d, "episode_id", None) == episode_id
         and getattr(d, "step", None) == step),
        None
    )
    if snap is None:
        print(f"[ERRORE] Nessuno snapshot trovato per episodio={episode_id}, step={step}")
        return

    # Ricostruisci grafo e calcola grandezze topologiche
    G, coords = build_nx_graph_from_wntr(wn, results)
    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)

    f = construct_matrix_f(wn, results)
    f_polygons = compute_polygon_flux(f, B2, abs=False)
    f_polygons_abs = compute_polygon_flux(f, B2, abs=True)

    # Limiti per le scale colore
    vmin_p, vmax_p = get_inital_polygons_flux_limits(f_polygons)
    vmin_n, vmax_n = get_initial_node_demand_limits(G)
    vmin_e, vmax_e = get_initial_edge_flow_limits(f)

    # Individua il nodo di leak dallo snapshot (etichetta y=1)
    leak_idx = (snap.y.squeeze() == 1).nonzero(as_tuple=True)[0]
    leak_node = None
    if len(leak_idx) > 0:
        leak_node_name = list(G.nodes())[int(leak_idx[0])]
        leak_node = wn.get_node(leak_node_name)
        print(f"[INFO] Leak al nodo: {leak_node_name}")

    print(f"\n Visualizzazione episodio={episode_id}, step={step}")
    plot_node_demand(G, coords, vmin_n, vmax_n, episode=episode_id, step=step)
    plot_edge_flowrate(G, coords, f, vmin_e, vmax_e, episode=episode_id, step=step)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin_p, vmax_p, leak_node, episode=episode_id, step=step)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons_abs, vmin_p, vmax_p, leak_node, episode=episode_id, step=step)

