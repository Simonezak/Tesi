import os
import torch
import torch.nn as nn
import numpy as np

from wntr_exp_multi import WNTREnv, build_static_graph_from_wntr
from GGNN_multi import GRUCell   # usa il tuo GRUCell già definito


# ============================================================
#  GGNN ENCODER (SOLO EMBEDDING SPAZIALE)
# ============================================================

class GGNNEncoder(nn.Module):
    def __init__(self, attr_size, hidden_size, propag_steps):
        super().__init__()

        self.hidden_size = hidden_size
        self.propag_steps = propag_steps

        self.linear_i = nn.Linear(attr_size, hidden_size)
        self.gru = GRUCell(2 * hidden_size, hidden_size)

    def forward(self, attr_matrix, adj_matrix):
        """
        attr_matrix: [1, N, F]
        adj_matrix:  [1, N, N]
        return:      [1, N, hidden_size]
        """

        A_in  = adj_matrix.float()
        A_out = adj_matrix.float().transpose(-2, -1)

        h = self.linear_i(attr_matrix).relu()

        for _ in range(self.propag_steps):
            a_in  = torch.bmm(A_in, h)
            a_out = torch.bmm(A_out, h)
            h = self.gru(torch.cat((a_in, a_out), dim=-1), h)

        return h


# ============================================================
#  GGNN + NODE-LSTM (LOCALIZZAZIONE LEAK)
# ============================================================

class GGNN_NodeLSTM_Localizer(nn.Module):
    def __init__(
        self,
        ggnn_encoder,
        hidden_size,
        lstm_hidden=128,
        lstm_layers=1
    ):
        super().__init__()

        self.ggnn = ggnn_encoder

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, attr_seq, adj):
        """
        attr_seq: list[T] of [1, N, 1]
        adj:      [1, N, N]
        return:   scores [N]
        """

        node_embeddings_time = []

        for attr in attr_seq:
            h = self.ggnn(attr, adj)       # [1, N, hidden]
            node_embeddings_time.append(h.squeeze(0))  # [N, hidden]

        # [T, N, hidden] -> [N, T, hidden]
        H = torch.stack(node_embeddings_time, dim=0).permute(1, 0, 2)

        lstm_out, _ = self.lstm(H)          # [N, T, lstm_hidden]
        z = lstm_out[:, -1, :]              # [N, lstm_hidden]

        scores = self.fc(z).squeeze(-1)     # [N]
        return scores


# ============================================================
#  TRAINING
# ============================================================

def train_and_save_model(inp_path):

    # -------------------------
    # CONFIG
    # -------------------------
    MAX_STEPS = 50
    NUM_EPISODES = 100
    TEMP_WINDOW = 6
    AREA = 0.1

    GGNN_HIDDEN = 132
    GGNN_PROPAG = 7
    LSTM_HIDDEN = 64

    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SAVE_DIR = "saved_models"
    MODEL_NAME = "ggnn_encoder_node_lstm.pt"

    # -------------------------
    # ENVIRONMENT
    # -------------------------
    env = WNTREnv(inp_path, max_steps=MAX_STEPS)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    # -------------------------
    # MODELS
    # -------------------------
    ggnn_encoder = GGNNEncoder(
        attr_size=1,
        hidden_size=GGNN_HIDDEN,
        propag_steps=GGNN_PROPAG
    ).to(DEVICE)

    model = GGNN_NodeLSTM_Localizer(
        ggnn_encoder=ggnn_encoder,
        hidden_size=GGNN_HIDDEN,
        lstm_hidden=LSTM_HIDDEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    print("\n=== TRAINING GGNN ENCODER + NODE LSTM ===")

    for ep in range(NUM_EPISODES):

        env.reset(num_leaks=1)
        sim = env.sim

        pressure_seq = []
        train_samples = []

        for step in range(MAX_STEPS):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=AREA)

            sim.step_sim()
            results = sim.get_results()

            pressures = torch.tensor(
                results.node["pressure"]
                .iloc[-1][list(node2idx.keys())]
                .values,
                dtype=torch.float32,
                device=DEVICE
            )

            # aggiungi SEMPRE lo step
            pressure_seq.append(pressures)

            # prima del leak non alleni
            if step < env.leak_start_step:
                continue

            # costruisci attr_seq con T variabile
            attr_seq = [
                p.view(1, -1, 1) for p in pressure_seq
            ]

            # target binario multi-leak
            target = torch.zeros(len(node2idx), device=DEVICE)
            for ln in env.leak_node_names:
                target[node2idx[ln]] = 1.0

            train_samples.append((attr_seq, target))


        # ---- optimization step
        for attr_seq, target in train_samples:
            optimizer.zero_grad()
            scores = model(attr_seq, adj_matrix)
            loss = loss_fn(scores, target)
            loss.backward()
            optimizer.step()

        print(f"Episodio {ep+1}/{NUM_EPISODES} | loss={loss.item():.4f}")

    # =====================================================
    # SAVE MODEL
    # =====================================================
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node2idx": node2idx,
            "config": {
                "ggnn_hidden": GGNN_HIDDEN,
                "ggnn_propag": GGNN_PROPAG,
                "lstm_hidden": LSTM_HIDDEN,
                "temp_window": TEMP_WINDOW
            }
        },
        save_path
    )

    print(f"\n✅ Modello salvato in: {save_path}")


# ============================================================
if __name__ == "__main__":

    train_and_save_model(
        inp_path="/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"
    )
