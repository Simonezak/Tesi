import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from wntr.network.elements import LinkStatus

import torch
from torch_geometric.data import Data


@dataclass
class GraphFeatureConfig:
    node_features: Tuple[str, ...] = ("elevation", "demand", "pressure", "leak_demand")
    edge_features: Tuple[str, ...] = ("length", "diameter", "flowrate", "headloss")
    include_only_junctions: bool = True   # Se True: includi solo i nodi "junction"
    undirected: bool = True               # Se True: aggiungi archi inversi


def run_wntr_simulation(inp_path: str,
                        simulation_duration: Optional[int] = None,
                        timestep_index: int = -1):
    import wntr
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"INP file non trovato: {inp_path}")
    wn = wntr.network.WaterNetworkModel(inp_path)
    if simulation_duration is not None:
        wn.options.time.duration = simulation_duration
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    time_index = results.time
    if len(time_index) == 0:
        raise RuntimeError("Nessun timestep prodotto dalla simulazione")
    if timestep_index < 0 or timestep_index >= len(time_index):
        timestep_index = len(time_index) - 1  # sempre ultimo valido
    return wn, results, timestep_index


def safe_get(df, timestep_index, col, default=0.0):
    if df is None or col not in df.columns or len(df) == 0:
        return default
    if timestep_index < 0:
        timestep_index = len(df) - 1
    elif timestep_index >= len(df):
        timestep_index = len(df) - 1
    return float(df.iloc[timestep_index][col])


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

        u, v = node2idx[u_name], node2idx[v_name]
        length = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow = safe_get(df_flow, timestep_index, pipe_name)
        headloss = safe_get(df_headloss, timestep_index, pipe_name)

        # forward
        forward_idx = len(edge_index_list)
        edge_index_list.append((u, v))
        edge_attrs.append([length, diameter, flow, headloss])
        edge_names.append(pipe_name)
        forward_edge_idx_for_pipe.append(forward_idx)
        pipe_names.append(pipe_name)

        # reverse (opzionale)
        if cfg.undirected:
            edge_index_list.append((v, u))
            edge_attrs.append([length, diameter, flow, headloss])
            edge_names.append(f"{pipe_name}__rev")

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

    return data, node2idx, idx2node, edge2idx, idx2edge
