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
    node_features: Tuple[str, ...] = ("elevation", "pressure")
    edge_features: Tuple[str, ...] = ("length", "diameter", "flowrate")
    undirected: bool = False



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
    junction_names: List[str] = []

    for name, node in wn.nodes():
        if isinstance(node, wntr.network.elements.Junction):
            junction_names.append(name)

    # Mappa nome → indice PyG
    node2idx = {name: i for i, name in enumerate(junction_names)}
    idx2node = {i: name for name, i in node2idx.items()}
    N = len(junction_names)

    # Lettura attributi di ogni nodo
    elev, demand, pressure, leak_dem = [], [], [], []

    df_demand: Optional[pd.DataFrame]   = results.node.get("demand", None)
    df_pressure: Optional[pd.DataFrame] = results.node.get("pressure", None)
    df_leak: Optional[pd.DataFrame]     = results.node.get("leak_demand", None)

    for name in junction_names:
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


    for pipe_name in wn.pipe_name_list:       # funziona anche con pump/valves se vuoi aggiungere
        pipe = wn.get_link(pipe_name)
        u_name = pipe.start_node_name
        v_name = pipe.end_node_name

        # 🔥 Escludi archi che toccano serbatoi/reservoir
        if u_name not in node2idx or v_name not in node2idx:
            continue

        u = node2idx[u_name]
        v = node2idx[v_name]

        length   = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow     = safe_get(df_flow, pipe_name)

        # aggiungi forward edge
        edge_index_list.append((u, v))
        edge_attrs.append([length, diameter, flow])

        # aggiungi backward edge (grafo non orientato oppure orientato 2-way)
        edge_index_list.append((v, u))
        edge_attrs.append([length, diameter, flow])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32)

    data = Data(
        x=x_torch,                # [N, 4]
        edge_index=edge_index,    # [2, E]
        edge_attr=edge_attr       # [E, 3]
    )

    data.node2idx = node2idx
    data.idx2node = idx2node

    return data, node2idx, idx2node



def build_static_graph_from_wntr(wn):
    """
    Costruisce:
    - adjacency matrix [1, N, N]
    - node2idx, idx2node
    usando SOLO le junctions.
    """

    # --- seleziona solo junctions ---
    junction_names = [
        name for name, node in wn.nodes()
        if isinstance(node, wntr.network.elements.Junction)
    ]

    node2idx = {name: i for i, name in enumerate(junction_names)}
    idx2node = {i: name for name, i in node2idx.items()}
    N = len(junction_names)

    # --- adjacency matrix ---
    adj = torch.zeros((N, N), dtype=torch.float32)

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        u_name = pipe.start_node_name
        v_name = pipe.end_node_name

        # scarta pipe che toccano tank / reservoir
        if u_name not in node2idx or v_name not in node2idx:
            continue

        u = node2idx[u_name]
        v = node2idx[v_name]

        adj[u, v] = 1.0
        adj[v, u] = 1.0   # grafo non orientato

    # aggiungi dimensione batch
    adj_matrix = adj.unsqueeze(0)  # [1, N, N]

    return adj_matrix, node2idx, idx2node


