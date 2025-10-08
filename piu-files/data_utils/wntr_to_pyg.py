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


@dataclass
class GraphFeatureConfig:
    node_features: Tuple[str, ...] = ("elevation", "demand", "pressure", "leak_demand")
    edge_features: Tuple[str, ...] = ("length", "diameter", "flowrate", "headloss")
    include_only_junctions: bool = True
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

    # ---- nodi ----
    node_names: List[str] = list(wn.junction_name_list) if cfg.include_only_junctions else list(wn.node_name_list)
    node2idx = {name: i for i, name in enumerate(node_names)}
    idx2node = {i: name for name, i in node2idx.items()}

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

    # ---- archi (pipes) ----
    edge_index_list: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    edge_names: List[str] = []

    forward_edge_idx_for_pipe: List[int] = []  # mapping pipe_id -> edge_index (direzione forward)
    pipe_names: List[str] = []

    df_flow = results.link.get("flowrate", None)
    df_headloss = results.link.get("headloss", None)

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)

        u_name, v_name = pipe.start_node_name, pipe.end_node_name
        if u_name not in node2idx or v_name not in node2idx:
            continue
        
        # per provare i tubi chiusi
        #if int(pipe_name) % 2 == 0:
        #    pipe.initial_status = LinkStatus.Closed

        #print(f"  • Pipe {pipe_name:>6s} → {pipe.initial_status}")

        # 🔹 controlla stato della pipe
        if pipe.initial_status != LinkStatus.Open:
            continue  # se chiusa → non aggiungere archi

        u, v = node2idx[u_name], node2idx[v_name]
        length = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow = safe_get(df_flow, timestep_index, pipe_name)
        headloss = safe_get(df_headloss, timestep_index, pipe_name)

        # forward edge
        forward_idx = len(edge_index_list)
        edge_index_list.append((u, v))
        edge_attrs.append([length, diameter, flow, headloss])
        edge_names.append(pipe_name)
        forward_edge_idx_for_pipe.append(forward_idx)
        pipe_names.append(pipe_name)

        # reverse edge (se cfg.undirected)
        if cfg.undirected:
            edge_index_list.append((v, u))
            edge_attrs.append([length, diameter, flow, headloss])
            edge_names.append(f"{pipe_name}__rev")
    
    print("chiamato build pyg")

    edge_index = torch.tensor(np.array(edge_index_list, dtype=np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = x.shape[0]

    # --- campi extra per azioni per-pipe ---
    data.pipe_edge_idx = torch.tensor(forward_edge_idx_for_pipe, dtype=torch.long)   # (P,)
    data.num_pipes = int(len(forward_edge_idx_for_pipe))
    data.pipe_names = pipe_names                                                    # list python

    # maschera stato attuale (1=open, 0=closed) per ogni pipe
    status_list = []
    for name in pipe_names:
        s = wn.get_link(name).initial_status
        status_list.append(1.0 if s == LinkStatus.Open else 0.0)
    data.pipe_open_mask = torch.tensor(status_list, dtype=torch.float32)            # (P,)

    edge2idx = {name: i for i, name in enumerate(edge_names)}
    idx2edge = {i: name for name, i in edge2idx.items()}

    #plot_current_network(wn, results, timestep_index)

    return data, node2idx, idx2node, edge2idx, idx2edge


def plot_current_network(wn, results, timestep_index, show_names=False):
    """
    Mostra la rete idrica attuale:
    - Nodi colorati per pressione
    - Tubi chiusi indicati da una X rossa al centro
    (nessun colore custom, massima compatibilità)
    """
    print(results.node["pressure"])

    pressures = results.node["pressure"].iloc[timestep_index]

    plt.figure(figsize=(9, 7))

    # 🔹 Disegna la rete base (senza colorare i link)
    wntr.graphics.network.plot_network(
        wn,
        node_attribute=pressures,
        node_size=40,
        node_range=[pressures.min(), pressures.max()],
        add_colorbar=True,
    )

    # 🔹 Disegna una X rossa al centro di ciascun tubo chiuso
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        if pipe.initial_status != LinkStatus.Open:
            u, v = pipe.start_node, pipe.end_node
            x_mid = (u.coordinates[0] + v.coordinates[0]) / 2
            y_mid = (u.coordinates[1] + v.coordinates[1]) / 2
            plt.scatter(x_mid, y_mid, color="red", marker="x", s=100, zorder=5)
            if show_names:
                plt.text(x_mid + 5, y_mid + 5, pipe_name, color="red",
                         fontsize=8, zorder=6)

    plt.title(f"Rete idrica al timestep {timestep_index}\n(rossa X = tubo chiuso)")
    plt.axis("equal")
    plt.show()

