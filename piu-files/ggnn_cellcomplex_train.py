# -*- coding: utf-8 -*-
"""
GGNN + Cell-Complex layer (B1/B2) + training script.

Questo file integra:
  1) la GGNN (stile GGNN_Regression.py) con input a finestra temporale
  2) un layer topologico su cell complex basato su B1 (nodi→archi) e B2 (archi→celle)
     calcolati con main_dyn_topologyknown_01.func_gen_B2_lu.

Come input resta IDENTICO alla tua GGNN con window:
  attr_matrix: [1, N, W]  (W = window_size)
  adj_matrix : [1, N, N]

Il layer cell-complex lavora sugli embedding prodotti dalla GGNN:
  h: [B, N, H] -> topological MP -> h': [B, N, H] -> head -> out: [B, N]

Esecuzione:
  python ggnn_cellcomplex_train.py

Modifica in fondo (main) il path .inp e gli iperparametri.

Nota:
- Se la rete è “branched” (pochi/nessun ciclo), B2 può venire vuota (C=0). In quel caso
  il layer fa comunque node↔edge tramite B1, ma la parte “celle” non contribuisce.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import wntr

# ---- Topology (B1, B2) ----
from main_dyn_topologyknown_01 import func_gen_B2_lu

# ---- Simulator (fallback import) ----
try:
    from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
except Exception:
    # nel repo spesso è presente come file locale
    from interactive_network_simulator import InteractiveWNTRSimulator


# ============================================================
#  Utils: adjacency + NX graph in indexing coerente
# ============================================================

def build_static_graph_from_wntr(
    wn: wntr.network.WaterNetworkModel
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Costruisce adj_matrix (1,N,N) e mapping nodi coerenti con wn.nodes().
    (Qui uso solo pipes; se vuoi includere anche pumps/valves, estendi i link list.)
    """
    node_names = [name for name, _ in wn.nodes()]
    node2idx = {name: i for i, name in enumerate(node_names)}
    idx2node = {i: name for name, i in node2idx.items()}

    n = len(node_names)
    adj = torch.zeros((n, n), dtype=torch.float32)

    for link_name in wn.pipe_name_list:
        link = wn.get_link(link_name)
        u_name, v_name = link.start_node_name, link.end_node_name
        if u_name not in node2idx or v_name not in node2idx:
            continue
        u, v = node2idx[u_name], node2idx[v_name]
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    return adj.unsqueeze(0), node2idx, idx2node


def build_nx_graph_from_wntr_indices(
    wn: wntr.network.WaterNetworkModel,
    node2idx: Dict[str, int]
) -> nx.Graph:
    """Grafo NX con nodi indicizzati 0..N-1 coerente con node2idx."""
    G = nx.Graph()
    for name, i in node2idx.items():
        G.add_node(i, name=name)

    for link_name in wn.pipe_name_list:
        link = wn.get_link(link_name)
        u_name, v_name = link.start_node_name, link.end_node_name
        if u_name not in node2idx or v_name not in node2idx:
            continue
        G.add_edge(node2idx[u_name], node2idx[v_name], link_name=link_name)

    return G


def dense_to_sparse_coo(mat: torch.Tensor) -> torch.Tensor:
    """Converte un tensore denso 2D in sparse COO."""
    assert mat.dim() == 2
    idx = mat.nonzero(as_tuple=False).t()
    vals = mat[idx[0], idx[1]]
    return torch.sparse_coo_tensor(
        idx, vals, mat.size(), dtype=mat.dtype, device=mat.device
    ).coalesce()


# ============================================================
#  Cell-Complex Layer (B1/B2) — minimal topological MP
# ============================================================

class CellComplexLayer(nn.Module):
    """
    Propagazione node↔edge↔cell con B1/B2 (sparse).

    Input:
      h: [B, N, H]

    Internamente:
      e  = B1^T h            [B, E, H]
      c  = B2^T e            [B, C, H]   (se C>0)
      e2 = B2 c              [B, E, H]
      h_msg = B1 (e + e2)    [B, N, H]

    Output:
      h_out: [B, N, H]
    """

    def __init__(
        self,
        hidden_dim: int,
        B1: torch.Tensor,                  # sparse (N,E)
        B2: Optional[torch.Tensor] = None,  # sparse (E,C) oppure None
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("B1", B1.coalesce())
        self.register_buffer("B1T", B1.transpose(0, 1).coalesce())

        self.has_cells = B2 is not None and B2.numel() > 0 and B2.size(1) > 0
        if self.has_cells:
            self.register_buffer("B2", B2.coalesce())
            self.register_buffer("B2T", B2.transpose(0, 1).coalesce())
        else:
            self.B2 = None
            self.B2T = None

        # Normalizzazione tramite gradi su |B|
        B1_abs = self.B1.abs()
        self.register_buffer("deg_nodes", torch.clamp(torch.sparse.sum(B1_abs, dim=1).to_dense(), min=1.0))  # [N]
        self.register_buffer("deg_edges", torch.clamp(torch.sparse.sum(B1_abs, dim=0).to_dense(), min=1.0))  # [E]

        if self.has_cells:
            B2_abs = self.B2.abs()
            self.register_buffer("deg_cells", torch.clamp(torch.sparse.sum(B2_abs, dim=0).to_dense(), min=1.0))             # [C]
            self.register_buffer("deg_edges_from_cells", torch.clamp(torch.sparse.sum(B2_abs, dim=1).to_dense(), min=1.0))  # [E]
        else:
            self.deg_cells = None
            self.deg_edges_from_cells = None

        act_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
        }
        self.act = act_map.get(activation.lower(), nn.ReLU())

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Gate residuo (quanto usare il messaggio topologico)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    @staticmethod
    def _spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """A sparse [m,n], X dense [n,H] -> [m,H]"""
        return torch.sparse.mm(A, X)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected h [B,N,H], got {tuple(h.shape)}")

        B, N, H = h.shape
        outs = []

        for b in range(B):
            hb = h[b]  # [N,H]

            # Nodes -> Edges
            e = self._spmm(self.B1T, hb)                  # [E,H]
            e = e / self.deg_edges.unsqueeze(-1)
            e = self.edge_mlp(self.act(e))
            e = self.dropout(e)

            # Edges -> Cells -> Edges (optional)
            if self.has_cells:
                c = self._spmm(self.B2T, e)               # [C,H]
                c = c / self.deg_cells.unsqueeze(-1)
                c = self.cell_mlp(self.act(c))
                c = self.dropout(c)

                e2 = self._spmm(self.B2, c)               # [E,H]
                e2 = e2 / self.deg_edges_from_cells.unsqueeze(-1)
            else:
                e2 = torch.zeros_like(e)

            # Back to nodes
            h_msg = self._spmm(self.B1, (e + e2))          # [N,H]
            h_msg = h_msg / self.deg_nodes.unsqueeze(-1)
            h_msg = self.node_mlp(self.act(h_msg))
            h_msg = self.dropout(h_msg)

            z = self.gate(torch.cat([hb, h_msg], dim=-1))
            hb_out = (1 - z) * hb + z * (hb + h_msg)      # residual + gated
            outs.append(hb_out)

        return torch.stack(outs, dim=0)  # [B,N,H]


# ============================================================
#  GGNN (come la tua) + encoder + head
# ============================================================

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input_: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        inputs_and_prev_state = torch.cat((input_, hidden_state), dim=-1)
        update_gate = self.linear_z(inputs_and_prev_state).sigmoid()
        reset_gate = self.linear_r(inputs_and_prev_state).sigmoid()
        new_hidden_state = self.linear(torch.cat((input_, reset_gate * hidden_state), -1)).tanh()
        output = (1 - update_gate) * hidden_state + update_gate * new_hidden_state
        return output


class GGNNEncoder(nn.Module):
    """Stessa GGNN di GGNN_Regression.py ma ritorna l'embedding dei nodi."""
    def __init__(self, attr_size: int, hidden_size: int, propag_steps: int):
        super().__init__()
        self.linear_i = nn.Linear(attr_size, hidden_size)
        self.gru = GRUCell(2 * hidden_size, hidden_size)
        torch.nn.init.kaiming_normal_(self.linear_i.weight)
        torch.nn.init.constant_(self.linear_i.bias, 0)

        self.propag_steps = propag_steps

    def forward(self, attr_matrix: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        A_in = adj_matrix.float()
        A_out = adj_matrix.float().transpose(-2, -1)

        if len(A_in.shape) < 3:
            A_in = A_in.unsqueeze(0)
            A_out = A_out.unsqueeze(0)
        if len(attr_matrix.shape) < 3:
            attr_matrix = attr_matrix.unsqueeze(0)

        h = self.linear_i(attr_matrix).relu()

        for _ in range(self.propag_steps):
            a_in = torch.bmm(A_in, h)
            a_out = torch.bmm(A_out, h)
            h = self.gru(torch.cat((a_in, a_out), dim=-1), h)

        return h  # [B,N,H]


class GGNNCellComplexModel(nn.Module):
    """GGNN encoder + CellComplexLayer + node-wise regression head."""
    def __init__(
        self,
        attr_size: int,
        hidden_size: int,
        propag_steps: int,
        B1: torch.Tensor,                  # sparse (N,E)
        B2: Optional[torch.Tensor] = None,  # sparse (E,C) oppure None
        topo_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = GGNNEncoder(attr_size, hidden_size, propag_steps)
        self.topo = CellComplexLayer(hidden_size, B1=B1, B2=B2, dropout=topo_dropout)
        self.readout = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_normal_(self.readout.weight)
        torch.nn.init.constant_(self.readout.bias, 0)

    def forward(self, attr_matrix: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        h = self.encoder(attr_matrix, adj_matrix)      # [B,N,H]
        h = self.topo(h)                               # [B,N,H]
        out = self.readout(h).squeeze(-1)              # [B,N]
        return out


# ============================================================
#  Environment / data generation
# ============================================================

class WNTREnv:
    def __init__(self, inp_path: str, max_steps: int = 50, hydraulic_timestep: int = 3600, num_leaks: int = 2):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.num_leaks = num_leaks
        self.wn: wntr.network.WaterNetworkModel = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = None
        self.leak_node_names: List[str] = []
        self.leak_start_step: int = 0

    def reset(self, with_leak: bool = True):
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

        self.leak_node_names = []
        if with_leak:
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]
            num = min(self.num_leaks, len(junctions))
            self.leak_node_names = np.random.choice(junctions, size=num, replace=False).tolist()
            self.leak_start_step = int(np.random.randint(5, min(26, self.max_steps - 1)))
            print(f"[LEAK] Nodi selezionati: {self.leak_node_names}")
            print(f"[LEAK] Leak start step: {self.leak_start_step}")
        else:
            print("[INIT] Episodio senza leak")

        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )


# ============================================================
#  Training
# ============================================================

@dataclass
class TrainConfig:
    inp_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulation
    num_episodes: int = 100
    max_steps: int = 50
    num_leaks: int = 2
    leak_area: float = 0.1

    # Model
    window_size: int = 4
    hidden_size: int = 132
    propag_steps: int = 7
    topo_dropout: float = 0.24509566542543507

    # Topology
    max_cycle_length: int = 5

    # Optimization
    epochs: int = 500
    lr: float = 1e-2
    seed: int = 42

    # Saving
    save_dir: str = "saved_models"
    save_name: str = "ggnn_cellcomplex.pt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_B1_B2_for_wn(
    wn: wntr.network.WaterNetworkModel,
    node2idx: Dict[str, int],
    max_cycle_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Calcola B1 (N,E) e B2 (E,C) usando func_gen_B2_lu.
    Ritorna sparse COO.
    """
    G = build_nx_graph_from_wntr_indices(wn, node2idx)

    # signature: func_gen_B2_lu(G, max_cycle_length)
    B1_np, B2_np, _ = func_gen_B2_lu(G, max_cycle_length=max_cycle_length) \
        if "max_cycle_length" in func_gen_B2_lu.__code__.co_varnames else func_gen_B2_lu(G, max_cycle_length)

    B1 = torch.tensor(B1_np, dtype=torch.float32, device=device)
    B1_sp = dense_to_sparse_coo(B1)

    if B2_np.size > 0 and B2_np.shape[1] > 0:
        B2 = torch.tensor(B2_np, dtype=torch.float32, device=device)
        B2_sp = dense_to_sparse_coo(B2)
    else:
        B2_sp = None

    return B1_sp, B2_sp


def train(cfg: TrainConfig) -> str:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = WNTREnv(cfg.inp_path, max_steps=cfg.max_steps, num_leaks=cfg.num_leaks)

    # Static graph
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(device)

    # Topology once
    B1_sp, B2_sp = compute_B1_B2_for_wn(env.wn, node2idx, cfg.max_cycle_length, device=device)

    # Dataset
    all_samples: List[Dict[str, torch.Tensor]] = []

    for ep in range(cfg.num_episodes):
        print(f"\n--- Episodio {ep+1}/{cfg.num_episodes} ---")
        env.reset(with_leak=True)

        for step in range(cfg.max_steps):
            if step == env.leak_start_step:
                for leak_node in env.leak_node_names:
                    env.sim.start_leak(leak_node, leak_area=cfg.leak_area, leak_discharge_coefficient=0.75)
            env.sim.step_sim()

        results = env.sim.get_results()
        df_pressure = results.node["pressure"]
        df_demand = results.node["demand"]
        df_leak = results.node.get("leak_demand", None)

        cols = list(node2idx.keys())
        P = df_pressure[cols].to_numpy(dtype=np.float32)  # [T,N]
        D = df_demand[cols].to_numpy(dtype=np.float32)    # [T,N]
        L = np.zeros_like(D) if df_leak is None else df_leak[cols].to_numpy(dtype=np.float32)

        T, N = P.shape

        for t in range(cfg.window_size - 1, T):
            if t < env.leak_start_step:
                continue  # usa solo step dopo onset

            window = P[t - cfg.window_size + 1: t + 1]  # [W,N]
            attr_matrix = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)  # [1,N,W]

            u = D[t] + L[t]  # target continuo per nodo
            y = torch.tensor(u, dtype=torch.float32).unsqueeze(0)  # [1,N]

            all_samples.append({"attr": attr_matrix, "y": y})

    if len(all_samples) == 0:
        raise RuntimeError("Dataset vuoto: controlla max_steps/window_size/leak_start_step")

    # Model
    model = GGNNCellComplexModel(
        attr_size=cfg.window_size,
        hidden_size=cfg.hidden_size,
        propag_steps=cfg.propag_steps,
        B1=B1_sp,
        B2=B2_sp,
        topo_dropout=cfg.topo_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING GGNN + CellComplex ===")
    model.train()

    for epoch in range(cfg.epochs):
        sample = random.choice(all_samples)
        attr = sample["attr"].to(device)
        y = sample["y"].to(device)

        optimizer.zero_grad()
        out = model(attr, adj_matrix)  # [1,N]
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | Loss={loss.item():.8f}")

    # Save
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "window_size": cfg.window_size,
            "hidden_size": cfg.hidden_size,
            "propag_steps": cfg.propag_steps,
            "max_cycle_length": cfg.max_cycle_length,
            "topo_dropout": cfg.topo_dropout,
        },
        save_path
    )

    print(f"\n[OK] Modello salvato in: {save_path}")
    return save_path


if __name__ == "__main__":
    # Cambia qui il path al tuo .inp
    cfg = TrainConfig(
        inp_path=r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    )
    train(cfg)
