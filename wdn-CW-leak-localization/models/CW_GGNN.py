# =======================
# CW_GGNN.py
# =======================
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class TopoCycleResidualNodeAlpha(nn.Module):
    """
    SAFE CW layer: comportamento equivalente al modello "buono"
    - usa solo diag(L1)
    - niente edge-edge mixing
    - residuo piccolo e stabile
    """

    def __init__(self, B1_np, B2_np, hidden_dim, alpha_max=0.05):
        super().__init__()

        if isinstance(B1_np, torch.Tensor):
            B1 = B1_np.float()
        else:
            B1 = torch.tensor(B1_np, dtype=torch.float32)

        if isinstance(B2_np, torch.Tensor):
            B2 = B2_np.float()
        else:
            B2 = torch.tensor(B2_np, dtype=torch.float32)

        self.register_buffer("B1", B1)        # [N,E]
        self.register_buffer("B1_T", B1.t())  # [E,N]

        L1 = (B1.t() @ B1) + (B2 @ B2.t())    # [E,E]
        self.register_buffer("L1_diag", torch.diag(L1))  # <-- SOLO DIAG

        self.alpha_max = float(alpha_max)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.5))  # alpha iniziale molto piccolo

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.ReLU()

    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def forward(self, H):
        """
        H: [B, N, H]
        """
        # nodes -> edges
        E_feat = torch.einsum("en,bnh->beh", self.B1_T, H)  # [B,E,H]

        # diag(L1) * E_feat  (COME NEL MODELLO VECCHIO)
        E_topo = E_feat * self.L1_diag.view(1, -1, 1)

        # edges -> nodes
        dH = torch.einsum("ne,beh->bnh", self.B1, E_topo)

        dH = self.act(self.proj(dH))

        return H + self.alpha() * dH

class GGNNWithTopoAlpha(nn.Module):
    """
    forward(attr_matrix, adj_matrix) -> anomaly [N]
    """
    def __init__(self, ggnn, topo_node_layer: TopoCycleResidualNodeAlpha):
        super().__init__()
        self.ggnn = ggnn
        self.topo = topo_node_layer

    def forward(self, attr_matrix, adj_matrix):
        """
        attr_matrix: [B, N, F]   (F generico: pressione/flow/mask/imbalance/...)
        adj_matrix : [B, N, N] o [N, N]
        return     : anomaly [N]
        """
        if adj_matrix.dim() == 2:
            A = adj_matrix.unsqueeze(0)
        else:
            A = adj_matrix

        A_in  = A.float()
        A_out = A.transpose(-2, -1).float()

        hidden_state = self.ggnn.linear_i(attr_matrix).relu()  # [B,N,H]

        for _ in range(self.ggnn.propag_steps):
            a_in  = torch.bmm(A_in,  hidden_state)
            a_out = torch.bmm(A_out, hidden_state)
            hidden_state = self.ggnn.gru(torch.cat((a_in, a_out), dim=-1), hidden_state)

        hidden_state = self.topo(hidden_state)

        anomaly = self.ggnn.linear_o(hidden_state).squeeze(-1).squeeze(0)
        return anomaly