import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool


class EdgeAwareGNN(nn.Module):
    """
    Encoder GNN basato su NNConv:
    - Node input: 4 (elevation, demand, pressure, leak_demand)
    - Edge input: 4 (length, diameter, flowrate, headloss)
    """
    def __init__(self, node_in=4, edge_in=4, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []

        # Primo NNConv: node_in -> hidden
        edge_nn1 = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.ReLU(),
            nn.Linear(64, node_in * hidden),
        )
        conv1 = NNConv(in_channels=node_in, out_channels=hidden, nn=edge_nn1, aggr="mean")
        layers.append(conv1)

        # Altri strati
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

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = x
        for conv, ln in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = self.act(h)
            h = ln(h)
            h = self.dropout(h)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        g = global_mean_pool(h, batch)  # embedding del grafo
        return g


class DQNGNN(nn.Module):
    """
    Testa DQN sopra l'encoder GNN.
    Ritorna Q-values per ogni azione discreta.
    """
    def __init__(self, action_dim, node_in=4, edge_in=4, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = EdgeAwareGNN(node_in=node_in, edge_in=edge_in,
                                    hidden=hidden, num_layers=num_layers,
                                    dropout=dropout)
        self.q_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, data: Data):
        g = self.encoder(data.x, data.edge_index, data.edge_attr, getattr(data, "batch", None))
        return self.q_head(g)  # (B, action_dim)

    def sample_action(self, data: Data, epsilon: float):
        q_values = self.forward(data)
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, q_values.size(1), (1,)).item()
        else:
            return q_values.argmax().item()
