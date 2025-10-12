import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool
import torch.nn.functional as F
import numpy as np



class EdgeAwareGNN(nn.Module):
    def __init__(self, node_in=4, edge_in=3, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []

        edge_nn1 = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.ReLU(),
            nn.Linear(64, node_in * hidden),
        )
        conv1 = NNConv(in_channels=node_in, out_channels=hidden, nn=edge_nn1, aggr="mean")
        layers.append(conv1)

        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(edge_in, 128),
                nn.ReLU(),
                nn.Linear(128, hidden * hidden),
            )
            conv = NNConv(in_channels=hidden, out_channels=hidden, nn=edge_nn, aggr="mean")
            layers.append(conv)

        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in layers])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, x, edge_index, edge_attr, batch=None, return_node_emb=False):
        h = x
        for conv, ln in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = self.act(h)
            h = ln(h)
            h = self.dropout(h)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        g = global_mean_pool(h, batch)
        if return_node_emb:
            return h, g
        return g


class DQNGNN(nn.Module):
    """
    Azioni per-pipe: 2 per ciascun tubo (0=close, 1=open).
    Uscita: (1, 2 * P) per un singolo grafo.
    """
    def __init__(self, node_in=4, edge_in=3, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = EdgeAwareGNN(node_in=node_in, edge_in=edge_in,
                                    hidden=hidden, num_layers=num_layers,
                                    dropout=dropout)
        # testa node-based: combina i due nodi estremi del pipe
        self.node_pair_mlp = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.q_node_out = nn.Linear(hidden, 2)    # 2 azioni per pipe

        # testa global-based
        self.q_global_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.q_global_out = nn.Linear(hidden, 2)

    def forward(self, data: Data, debug: bool=False): # forward non crea il grafo ma lo riconosce solo. quindi il grafo è gia creato con build from wntr
        # encoder
        node_emb, global_emb = self.encoder(
            data.x, data.edge_index, data.edge_attr,
            getattr(data, "batch", None), return_node_emb=True
        )

        # mapping edge -> (u, v)
        edge_u = data.edge_index[0]
        edge_v = data.edge_index[1]

        # pipe forward edge indices
        pipe_edge_idx = getattr(data, "pipe_edge_idx", None)
        if pipe_edge_idx is None:
            raise ValueError("Manca data.pipe_edge_idx.")
        P = int(data.pipe_edge_idx.numel())

        valid_mask = data.pipe_edge_idx >= 0
        pipe_edge_idx_valid = data.pipe_edge_idx[valid_mask]

        pu = torch.zeros((P, node_emb.size(1)), device=node_emb.device)
        pv = torch.zeros((P, node_emb.size(1)), device=node_emb.device)

        if pipe_edge_idx_valid.numel() > 0:
            pu_valid = node_emb[edge_u[pipe_edge_idx_valid]]
            pv_valid = node_emb[edge_v[pipe_edge_idx_valid]]
            pu[valid_mask] = pu_valid
            pv[valid_mask] = pv_valid

        pair = torch.cat([pu, pv], dim=-1)            # (P, 2H)
        node_pair_feat = self.node_pair_mlp(pair)     # (P, H)
        q_node = self.q_node_out(node_pair_feat)      # (P, 2)

        print(f"P_tot={P}, valid={valid_mask.sum().item()}")


        # global per-graph (assumiamo un solo grafo alla volta)
        q_global = self.q_global_out(self.q_global_head(global_emb))  # (1, 1)

        # somma delle due teste
        q_per_pipe2 = q_node + q_global               # (P, 2)

        # mask azioni: (can_close, can_open)
        pipe_open = getattr(data, "pipe_open_mask", None)
        if pipe_open is None:
            pipe_open = torch.ones(P, device=q_per_pipe2.device)
        can_close = pipe_open.clone()                         # 1 se aperto
        can_open = 1.0 - pipe_open                    # 1 se chiuso
        action_mask = torch.stack([can_close, can_open], dim=-1)  # (P, 2)
        print(action_mask.shape)

        # maschera sui Q
        invalid = (action_mask < 0.5)
        q_masked = q_per_pipe2.masked_fill(invalid, -1e9)         # (P, 2)

        # questo è per evitare di fare close_all_pipes che non lascerebbe alcun arco 
        # nel grafo e quindi non si potrebbe piu costruire
        #q_actions = torch.cat([q_masked.reshape(1, -1), q_global], dim=1)  # (1, 2*P + 1)
        q_actions = q_masked.reshape(1, -1)  # (1, 2*P)


        if debug:
            print(f"node_emb: {tuple(node_emb.shape)}")               # (N, H)
            print(f"global_emb: {tuple(global_emb.shape)}")           # (1, H)
            print(f"q_node: {tuple(q_node.shape)}")                   # (P, 2)
            print(f"q_global: {tuple(q_global.shape)}")               # (P, 2)
            print(f"node_action_mask: {tuple(action_mask.shape)}")    # (P, 2)
            print(f"node_actions: {tuple(q_node.shape)}")             # alias
            print(f"global_actions: {tuple(q_global.shape)}")         # alias
            print(f"q_actions (flatten): {tuple(q_actions.shape)}")   # (1, 2*P)

        return q_actions

    def sample_action(self, data, global_step, temperature: float = 1.0):

        epsilon = compute_epsilon(global_step)
        print(epsilon)

        q_values = self.forward(data)  # i dati sono mandati a nn.Module. prende in input lo stato del mondo (il grafo PyG) e restituisce le Q-values per tutte le azioni possibili
        q_values = q_values.detach().cpu().squeeze()

        if torch.rand(1).item() < epsilon:
            # Esplorazione pesata: probabilità ∝ exp(Q / T)
            probs = F.softmax(q_values / temperature, dim=0)
            action = torch.multinomial(probs, 1).item()
            return action
        else:
            # Sfruttamento: azione migliore
            return q_values.argmax().item()

"""Decay esponenziale classico"""
def compute_epsilon(global_step, eps_start=1.0, eps_end=0.05, eps_decay=1e-1):
    return eps_end + (eps_start - eps_end) * np.exp(-eps_decay * global_step)
