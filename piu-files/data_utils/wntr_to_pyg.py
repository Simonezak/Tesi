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
    edge_features: Tuple[str, ...] = ("length", "diameter", "flowrate")
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
    from torch_geometric.data import Data
    from wntr.network.elements import Junction, LinkStatus

    # ---- nodi ----
    node_names: List[str] = []
    
    if cfg.include_only_junctions:
        node_names = [name for name, node in wn.nodes() if isinstance(node, Junction)]
    else:
        node_names = [name for name, _ in wn.nodes()]


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
    x_torch = torch.tensor(x, dtype=torch.float32)

    # salva dati nodi in CSV
    node_df = pd.DataFrame({
        "node_name": node_names,
        "elevation": elev,
        "demand": demand,
        "pressure": pressure,
        "leak_demand": leak_dem,
    })
    node_df.to_csv(f"graph_nodes.csv", index=False)

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

    # salva dati archi in CSV
    edge_df = pd.DataFrame({
        "pipe_name": pipe_names,
        "start_node": starts,
        "end_node": ends,
        "length": lengths,
        "diameter": diameters,
        "flow": flows,
        "status": statuses,
    })
    edge_df.to_csv(f"graph_edges.csv", index=False)

    # ---- costruisci Data PyG ----
    edge_index = torch.tensor(np.array(edge_index_list, dtype=np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float32)

    data = Data(x=x_torch, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = x.shape[0]

    # campi extra
    data.pipe_edge_idx = torch.tensor(forward_edge_idx_for_pipe, dtype=torch.long)
    data.num_pipes = len(forward_edge_idx_for_pipe)
    data.pipe_names = pipe_names

    # maschera stato (1=open, 0=closed)
    status_list = [1.0 if wn.get_link(name).status == LinkStatus.Open else 0.0 for name in pipe_names]
    data.pipe_open_mask = torch.tensor(status_list, dtype=torch.float32)

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
        if pipe.status != LinkStatus.Open:
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

