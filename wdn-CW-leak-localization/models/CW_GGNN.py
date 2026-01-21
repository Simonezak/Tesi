import torch
import torch.nn as nn


class TopoCycleResidualNodeAlpha(nn.Module):
    """
    Node-topology residual using cycle-aware edge Hodge Laplacian:
        L1 = B1^T B1 + B2 B2^T  (in edge space)
    and then pull back to nodes:
        dH = B1 ( L1 ( B1^T H ) )

    H_out = H + alpha * dH

    Alpha is constrained to be small via:
        alpha = alpha_max * sigmoid(alpha_raw)
    so it cannot explode and "modify too much".
    """
    def __init__(self, B1_np, B2_np, hidden_dim, alpha_max=0.2, use_layernorm=True):
        super().__init__()

        B1 = torch.tensor(B1_np, dtype=torch.float32)  # [N, E]
        B2 = torch.tensor(B2_np, dtype=torch.float32)  # [E, F]

        self.register_buffer("B1", B1)
        self.register_buffer("B1_T", B1.t())

        L1 = (B1.t() @ B1) + (B2 @ B2.t())            # [E, E]
        self.register_buffer("L1", L1)

        self.alpha_max = float(alpha_max)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12 => alpha ~ 0.024 if alpha_max=0.2

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # per monitorare quanto stai modificando (utile in training)
        self.last_delta_norm = None

    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def forward(self, H):
        """
        H: [B, N, Hdim]
        """
        # nodes -> edges
        E_feat = torch.einsum("en,bnh->beh", self.B1_T, H)     # [B, E, H]

        # cycle-aware filtering in edge space
        E_topo = torch.einsum("ee,beh->beh", self.L1, E_feat)  # [B, E, H]

        # edges -> nodes
        dH = torch.einsum("ne,beh->bnh", self.B1, E_topo)      # [B, N, H]

        dH = self.norm(self.act(self.proj(dH)))

        a = self.alpha()
        H_out = H + a * dH

        # salva misura (non cambia l'interfaccia)
        with torch.no_grad():
            self.last_delta_norm = (a * dH).pow(2).mean().sqrt().item()

        return H_out


class GGNNWithTopoAlpha(nn.Module):
    """
    Wrapper GGNN che aggiunge SOLO un contesto topologico residuale sui nodi
    (senza passare a embedding sugli archi come GGNNWithTopoMLP).

    Stessa signature e output della GGNNModel.forward:
        forward(attr_matrix, adj_matrix) -> anomaly [N]
    """
    def __init__(self, ggnn, topo_node_layer: TopoCycleResidualNodeAlpha):
        super().__init__()
        self.ggnn = ggnn
        self.topo = topo_node_layer

    def forward(self, attr_matrix, adj_matrix):
        """
        attr_matrix: [B, N, 1]
        adj_matrix : [B, N, N] o [N, N]
        return     : anomaly [N]
        """
        if adj_matrix.dim() == 2:
            A = adj_matrix.unsqueeze(0)
        else:
            A = adj_matrix

        A_in  = A.float()
        A_out = A.transpose(-2, -1).float()

        # hidden GGNN (identico al forward originale)
        hidden_state = self.ggnn.linear_i(attr_matrix).relu()  # [B,N,H]

        for _ in range(self.ggnn.propag_steps):
            a_in  = torch.bmm(A_in,  hidden_state)
            a_out = torch.bmm(A_out, hidden_state)
            hidden_state = self.ggnn.gru(torch.cat((a_in, a_out), dim=-1), hidden_state)

        # --- aggiunta topologica (residuale, controllata da alpha) ---
        hidden_state = self.topo(hidden_state)

        # output come GGNN
        anomaly = self.ggnn.linear_o(hidden_state).squeeze(-1).squeeze(0)
        return anomaly


