import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import wntr
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from wntr.network.elements import LinkStatus
from main_dyn_topologyknown_01 import func_gen_B2_lu
import networkx as nx


import numpy as np
import networkx as nx
import wntr

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
        status = 1.0 if pipe.status == wntr.network.elements.LinkStatus.Open else 0.0

        G.add_edge(
            start,
            end,
            pipe_name=pipe_name,
            length=length,
            diameter=diameter,
            flowrate=flow,
            open_mask=status,
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


def safe_get(df, timestep_index, col, default=0.0):
    """Legge in modo sicuro un valore da un DataFrame dei risultati (anche se parziale)."""
    if df is None or col not in df.columns or len(df) == 0:
        return default
    # Se non specificato, prende l’ultimo timestep disponibile
    try:
        return float(df.iloc[-1][col])
    except Exception:
        return default
    

def build_pyg_from_wntr(
    wn,
    results,
    timestep_index: int,
    cfg: GraphFeatureConfig = GraphFeatureConfig(),
):
    import numpy as np
    import pandas as pd
    from torch_geometric.data import Data
    from wntr.network.elements import Junction, LinkStatus

    # ---- nodi ----
    node_names: List[str] = [name for name, _ in wn.nodes()]  # includi TUTTI i nodi
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
        demand.append(safe_get(df_demand, timestep_index, name))
        pressure.append(safe_get(df_pressure, timestep_index, name))
        leak_dem.append(safe_get(df_leak, timestep_index, name))
        x = np.stack([elev, demand, pressure, leak_dem], axis=1)
        x_torch = torch.tensor(x, dtype=torch.float32)

    # ---- archi (pipes) ----
    edge_index_list: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    edge_names: List[str] = []

    forward_edge_idx_for_pipe: List[int] = []  # mapping pipe_id -> edge_index (direzione forward)
    pipe_names: List[str] = []

    df_flow = results.link.get("flowrate", None)

    lengths, diameters, flows, starts, ends, statuses = [], [], [], [], [], []


    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        u_name, v_name = pipe.start_node_name, pipe.end_node_name
        if u_name not in node2idx or v_name not in node2idx:
            continue

        # controlla stato (solo se pipe aperta)
        if pipe.status != LinkStatus.Open:
            continue
        status = 1.0 if pipe.status == LinkStatus.Open else 0.0

        u, v = node2idx[u_name], node2idx[v_name]
        length = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow = safe_get(df_flow, timestep_index, pipe_name)

        # aggiungi forward edge
        forward_idx = len(edge_index_list)
        edge_index_list.append((u, v))
        edge_attrs.append([length, diameter, flow])
        edge_names.append(pipe_name)
        forward_edge_idx_for_pipe.append(forward_idx)
        pipe_names.append(pipe_name)

        lengths.append(length)
        diameters.append(diameter)
        flows.append(flow)
        starts.append(u_name)
        ends.append(v_name)
        statuses.append(status)

        # se grafo non orientato, aggiungi anche reverse
        if cfg.undirected:
            edge_index_list.append((v, u))
            edge_attrs.append([length, diameter, flow])
            edge_names.append(f"{pipe_name}__rev")

    all_pipes = wn.pipe_name_list   
    pipe_edge_idx = []
    pipe_open_mask = []

    for pipe_name in all_pipes:
        pipe = wn.get_link(pipe_name)
        # Trova se il tubo compare nel grafo (solo se era aperto)
        if pipe_name in edge_names:
            idx = edge_names.index(pipe_name)
            pipe_edge_idx.append(idx)
            pipe_open_mask.append(1.0)  # tubo aperto e presente nel grafo
        else:
            pipe_edge_idx.append(-1)    # tubo chiuso, nessun arco nel grafo
            pipe_open_mask.append(0.0)  # 0 = chiuso, ma azione ancora possibile

    # ---- costruisci Data PyG ----
    edge_index = torch.tensor(np.array(edge_index_list, dtype=np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float32)

    data = Data(x=x_torch, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = x.shape[0]

    # campi extra
    data.pipe_edge_idx = torch.tensor(pipe_edge_idx, dtype=torch.long)
    data.pipe_open_mask = torch.tensor(pipe_open_mask, dtype=torch.float32)
    data.pipe_names = all_pipes
    data.num_pipes = len(forward_edge_idx_for_pipe)

    edge2idx = {name: i for i, name in enumerate(edge_names)}
    idx2edge = {i: name for name, i in edge2idx.items()}

    return data, node2idx, idx2node, edge2idx, idx2edge


def build_node_topo_features(G, B1, B2, f_polygons):
    """
    Restituisce topo_feats: array [N_nodes, K] con:
      - degree (grado del nodo in G)
      - cycle_influence (flusso poligonale proiettato sui nodi)
    Usa |B2| per proiettare flussi facce→spigoli e |B1|^T per spigoli→nodi.
    Funziona se B1 è (E x N) o (N x E): lo rileva da solo.
    """
    # --- shapes e valori assoluti per aggregazioni positive
    B1 = np.array(B1)
    B2 = np.array(B2)
    f_polygons = np.array(f_polygons).reshape(-1, 1)  # [F,1] oppure [num_cycles,1]

    # Determina dimensioni
    n_nodes = G.number_of_nodes()
    # edge_from_poly: [E,1] = |B2| @ |f_polygons|
    edge_from_poly = np.abs(B2) @ np.abs(f_polygons)

    # Proiezione su nodi: se B1 è (E x N), usiamo |B1|^T @ edge; se è (N x E), usiamo |B1| @ edge
    if B1.shape[0] == edge_from_poly.shape[0]:          # (E x N)
        node_cycle_flux = (np.abs(B1).T @ edge_from_poly).reshape(-1)
    elif B1.shape[1] == edge_from_poly.shape[0]:        # (N x E)
        node_cycle_flux = (np.abs(B1)   @ edge_from_poly).reshape(-1)
    else:
        raise ValueError(f"B1 shape {B1.shape} incompatible with edges={edge_from_poly.shape[0]}")

    # Controlla che la lunghezza sia giusta
    if node_cycle_flux.shape[0] != n_nodes:
        raise ValueError(f"node_cycle_flux len {node_cycle_flux.shape[0]} != n_nodes {n_nodes}")

    # Feature semplici di centralità topologica (opzionali, ma utili)
    deg = np.array([G.degree(n) for n in G.nodes()], dtype=float)

    # Stack finale: [N, K]
    topo_feats = np.stack([deg, node_cycle_flux], axis=1)
    return topo_feats  # numpy





