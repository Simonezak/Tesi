import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GNNLeakDetector(nn.Module):
    """
    Simple GNN for Leak Detection (node-level classification).
    Each node receives a probability of leak (0=no leak, 1=leak).
    """

    def __init__(self, node_in_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        # Layers
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        """
        Args:
            data (torch_geometric.data.Data): graph with
                - x : node features [num_nodes, node_in_dim]
                - edge_index : [2, num_edges]
        Returns:
            torch.Tensor : leak probabilities per node [num_nodes, 1]
        """
        x, edge_index = data.x, data.edge_index

        # 1st GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # 2nd GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Output (per-node leak probability)
        out = torch.sigmoid(self.linear_out(x))
        return out

    def predict(self, data: Data, threshold: float = 0.5):
        """
        Returns binary predictions for leak presence.
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(data)
            return (probs > threshold).float(), probs


import torch
import torch.nn as nn
import numpy as np

def train_model(model, optimizer, graphs, labels, loss_fn=None, epochs=50, name="model"):
    """
    Allena un modello GNN supervisionato su una lista di grafi PyG.

    Args:
        model: istanza della rete neurale PyTorch
        optimizer: ottimizzatore (es. Adam)
        graphs: lista di oggetti torch_geometric.data.Data
        labels: lista di tensori [num_nodes, 1]
        loss_fn: funzione di loss (default: BCELoss)
        epochs: numero di epoche
        name: stringa descrittiva (per logging)
    """
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()

        for data, y in zip(graphs, labels):
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % max(1, epochs // 10) == 0:
            print(f"[{name}] epoch {epoch:03d}/{epochs} | loss={total_loss / len(graphs):.4f}")


def eval_and_plot_at(
    step_idx,
    tag,
    model_plain,
    model_topo,
    graphs,
    aux,
    leak_name,
    plot_node_demand,
    plot_edge_flowrate,
    plot_cell_complex_flux,
    plot_leak_probability,
    wn,
    func_gen_B2_lu,
):
    """
    Valuta e plotta le predizioni di GCN e GCN+TopoLayer su un determinato step.

    Args:
        step_idx (int): indice dello step da valutare (0-based)
        tag (str): nome del test o descrizione
        model_plain: modello GCN semplice
        model_topo: modello GCN con Topological Layer
        graphs: lista di grafi PyG
        aux: lista con tuple (G, coords, results, B1, B2, f, f_polygons, node2idx, idx2node)
        leak_name (str): nome del nodo con leak reale
        plot_node_demand, plot_edge_flowrate, plot_cell_complex_flux, plot_leak_probability:
            funzioni di plotting importate da topological.py
        wn: modello WNTR
        func_gen_B2_lu: funzione per ricostruire i cicli topologici
    """
    print(f"\n=== [EVAL {tag}] step {step_idx + 1} ===")

    data = graphs[step_idx]
    G, coords, results, B1, B2, f, f_polygons, node2idx, idx2node = aux[step_idx]

    with torch.no_grad():
        p_plain = model_plain(data).squeeze()
        p_topo  = model_topo(data).squeeze()

    # --- Top 3 nodi più probabili ---
    values_plain, indices_plain = torch.topk(p_plain, k=3)
    values_topo, indices_topo   = torch.topk(p_topo,  k=3)

    top_plain = [(idx2node[i.item()], i.item(), v.item()) for i, v in zip(indices_plain, values_plain)]
    top_topo  = [(idx2node[i.item()], i.item(), v.item()) for i, v in zip(indices_topo,  values_topo)]

    print("[GCN]   Top-3:", ", ".join([f"{n} (idx {i}) → {v:.4f}" for n, i, v in top_plain]))
    print("[GCN+T] Top-3:", ", ".join([f"{n} (idx {i}) → {v:.4f}" for n, i, v in top_topo]))
    print(f"Leak reale: {leak_name} (idx {node2idx.get(leak_name, 'NA')})")

    # --- Plot probabilità ---
    try:
        plot_leak_probability(G, coords, p_plain, leak_node=node2idx[leak_name])
        plot_leak_probability(G, coords, p_topo,  leak_node=node2idx[leak_name])
    except Exception as e:
        print(f"[WARN plot_leak_probability] {e}")

    # --- Plot idraulici e topologici di contesto ---
    try:
        plot_node_demand(wn, results, step=step_idx + 1)
    except Exception as e:
        print(f"[WARN plot_node_demand] {e}")

    try:
        plot_edge_flowrate(wn, results, step=step_idx + 1)
    except Exception as e:
        print(f"[WARN plot_edge_flowrate] {e}")

    try:
        selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)[2]
        plot_cell_complex_flux(G, coords, selected_cycles, f_polygons=f_polygons)
    except Exception as e:
        print(f"[WARN plot_cell_complex_flux] {e}")
