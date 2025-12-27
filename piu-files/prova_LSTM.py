import torch
import torch.nn as nn
import numpy as np
import os

from wntr_exp_multi import WNTREnv, build_static_graph_from_wntr
from GGNN_multi import GGNNModel


# ============================================================
#  MODELLO: LSTM SU EMBEDDING GGNN
# ============================================================

class GGNNEmbeddingLSTM(nn.Module):
    """
    GGNN (embedding spaziale) + LSTM temporale (per nodo)
    """

    def __init__(
        self,
        ggnn: GGNNModel,
        lstm_hidden=64,
        lstm_layers=1
    ):
        super().__init__()

        self.ggnn = ggnn  # RIUSA IL TUO GGNN
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=1,        # u_t per nodo
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.linear_out = nn.Linear(lstm_hidden, 1)

    def forward(self, attr_seq, adj_matrix):
        """
        attr_seq: list di tensor [1, N, attr_size]
        adj_matrix: [1, N, N]

        ritorna:
            u_pred: [N]
        """

        ggnn_outputs = []

        # ---- GGNN per ogni timestep
        for attr in attr_seq:
            with torch.set_grad_enabled(self.ggnn.training):
                u_t = self.ggnn(attr, adj_matrix)   # [1, N]
            ggnn_outputs.append(u_t.squeeze(0))     # [N]

        # ---- Stack temporale
        # [T, N] -> [N, T, 1]
        H = torch.stack(ggnn_outputs, dim=0).transpose(0, 1).unsqueeze(-1)

        lstm_out, _ = self.lstm(H)                  # [N, T, lstm_hidden]
        z_T = lstm_out[:, -1, :]                    # [N, lstm_hidden]

        u_pred = self.linear_out(z_T).squeeze(-1)   # [N]
        return u_pred


# ============================================================
#  TRAINING
# ============================================================

def train_ggnn_embedding_lstm(inp_path):

    # -------------------------
    # Config
    # -------------------------
    MAX_STEPS = 50
    NUM_EPISODES = 100
    TEMP_WINDOW = 6
    AREA = 0.1

    GGNN_HIDDEN = 132
    GGNN_PROPAG = 7
    LSTM_HIDDEN = 64

    LR = 1e-2
    EPOCHS = 300

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Environment
    # -------------------------
    env = WNTREnv(inp_path, max_steps=MAX_STEPS)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    # -------------------------
    # GGNN (UGUALE AL TUO)
    # -------------------------
    ggnn = GGNNModel(
        attr_size=1,                 # pressione singola
        hidden_size=GGNN_HIDDEN,
        propag_steps=GGNN_PROPAG
    ).to(DEVICE)

    # 🔒 Consigliato: congela il GGNN
    for p in ggnn.parameters():
        p.requires_grad = False

    # -------------------------
    # Modello completo
    # -------------------------
    model = GGNNEmbeddingLSTM(
        ggnn=ggnn,
        lstm_hidden=LSTM_HIDDEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    loss_fn = nn.MSELoss()

    # =====================================================
    # TRAIN
    # =====================================================
    print("\n=== TRAINING GGNN-Embedding + LSTM ===")

    for ep in range(NUM_EPISODES):

        env.reset(with_leak=True)
        sim = env.sim

        pressure_buffer = []
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

            pressure_buffer.append(pressures)
            if len(pressure_buffer) > TEMP_WINDOW:
                pressure_buffer.pop(0)

            if len(pressure_buffer) < TEMP_WINDOW:
                continue
            if step < env.leak_start_step:
                continue

            # ---- costruisci sequenza attr per GGNN
            attr_seq = [
                p.view(1, -1, 1) for p in pressure_buffer
            ]

            demand = results.node["demand"].iloc[-1][list(node2idx.keys())].values
            leak = results.node.get("leak_demand", None)
            leak = leak.iloc[-1][list(node2idx.keys())].values if leak is not None else np.zeros_like(demand)

            target = torch.tensor(demand + leak, dtype=torch.float32, device=DEVICE)

            train_samples.append((attr_seq, target))

        # ---- update pesi
        for attr_seq, target in train_samples:
            optimizer.zero_grad()
            u_pred = model(attr_seq, adj_matrix)
            loss = loss_fn(u_pred, target)
            loss.backward()
            optimizer.step()

        print(f"Episodio {ep+1}/{NUM_EPISODES} completato")

    print("\n=== TRAINING COMPLETATO ===")

    SAVE_DIR = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\saved_models"
    MODEL_NAME = "ggnn_embedding_lstm.pt"

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "ggnn_hidden": GGNN_HIDDEN,
                "ggnn_propag": GGNN_PROPAG,
                "lstm_hidden": LSTM_HIDDEN,
                "temp_window": TEMP_WINDOW
            }
        },
        save_path
    )

    return model


# ============================================================
if __name__ == "__main__":

    train_ggnn_embedding_lstm(
        inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\20x20_branched.inp"
    )
