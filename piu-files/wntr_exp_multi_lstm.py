# =============================
#  wntr_exp_multi.py (LSTM version)
#  COMPLETELY REWRITTEN VERSION
# =============================

import torch
import torch.nn as nn
import numpy as np
import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator

from wntr_to_pyg import build_pyg_from_wntr
from GGNN_multi import GGNN_LSTM

# =============================
# PARAMETERS
# =============================
SEQ_LEN = 5            # length of temporal sequence
LR = 1e-3              # learning rate
EPOCHS = 200           # training epochs
NUM_EPISODES = 5       # training episodes
MAX_STEPS = 30         # steps per episode

# =============================
# ENVIRONMENT
# =============================
class WNTREnv:
    def __init__(self, inp_path, max_steps=MAX_STEPS, hydraulic_timestep=3600, num_leaks=2):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.num_leaks = num_leaks
        self.sim = None
        self.wn = None

    def reset(self, with_leak=True):
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

        # Leak nodes
        self.leak_node_names = []
        if with_leak:
            junctions = [name for name, node in self.wn.nodes()
                         if isinstance(node, wntr.network.elements.Junction)]
            self.leak_node_names = np.random.choice(junctions, size=min(self.num_leaks, len(junctions)), replace=False).tolist()
            self.leak_start_step = np.random.randint(5, 11)
            print(f"[LEAK] Nodes: {self.leak_node_names}")
            print(f"[LEAK] Leak starts at step {self.leak_start_step}")

        self.sim.init_simulation(global_timestep=self.hydraulic_timestep,
                                 duration=self.max_steps * self.hydraulic_timestep)

# =============================
# GGNN INPUT CONVERSION
# =============================
def pyg_to_ggnn_inputs(data):
    pressure = data.x[:, 2].view(1, -1, 1).float()

    N = data.num_nodes
    adj = torch.zeros((N, N), dtype=torch.float32)
    src = data.edge_index[0]
    dst = data.edge_index[1]
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0

    return pressure, adj.view(1, N, N)

# =============================
# MAIN TRAINING FUNCTION
# =============================
def run_GGNN(inp_path):

    env = WNTREnv(inp_path, max_steps=MAX_STEPS)
    all_sequences = []

    print("\n=== BUILDING TRAINING DATA ===")

    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep+1}/{NUM_EPISODES}")
        env.reset(with_leak=True)
        wn, sim = env.wn, env.sim

        attr_buffer = []
        adj_buffer = []

        for step in range(MAX_STEPS):
            if step == env.leak_start_step:
                for leak_node in env.leak_node_names:
                    sim.start_leak(leak_node, leak_area=0.09, leak_discharge_coefficient=0.75)

            sim.step_sim()
            results = sim.get_results()

            data, node2idx, idx2node, _, _ = build_pyg_from_wntr(wn, results)

            # Target u = demand + leak
            demand = data.x[:, 1]
            leak = data.x[:, 3]
            u = (demand + leak).float()

            # GGNN input
            attr, adj = pyg_to_ggnn_inputs(data)

            attr_buffer.append(attr)
            adj_buffer.append(adj)

            if len(attr_buffer) > SEQ_LEN:
                attr_buffer.pop(0)
                adj_buffer.pop(0)

            # Only store once we have a full sequence
            if len(attr_buffer) == SEQ_LEN:
                all_sequences.append({
                    "attr_seq": list(attr_buffer),
                    "adj_seq": list(adj_buffer),
                    "u": u,
                })

    # =============================
    # TRAIN GGNN-LSTM
    # =============================
    print("\n=== TRAINING GGNN-LSTM ===")

    model = GGNN_LSTM(attr_size=1, hidden_size=64, propag_steps=6,
                      lstm_hidden=64, lstm_layers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        sample = np.random.choice(all_sequences)

        attr_seq = sample["attr_seq"]
        adj_seq = sample["adj_seq"]
        target = sample["u"]   # [N]

        optimizer.zero_grad()
        out = model(attr_seq, adj_seq)    # [N]

        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss = {loss.item():.6f}")

    # =============================
    # TEST PHASE
    # =============================
    print("\n=== TEST PHASE ===")

    test_env = WNTREnv(inp_path, max_steps=MAX_STEPS)
    test_env.reset(with_leak=True)
    wn, sim = test_env.wn, test_env.sim

    test_attr_buf = []
    test_adj_buf = []
    anomaly_time_series = []

    for step in range(MAX_STEPS):
        if step == test_env.leak_start_step:
            for leak_node in test_env.leak_node_names:
                sim.start_leak(leak_node, leak_area=0.05, leak_discharge_coefficient=0.75)

        sim.step_sim()
        results = sim.get_results()

        data, node2idx, idx2node, _, _ = build_pyg_from_wntr(wn, results)

        demand = data.x[:, 1]
        leak = data.x[:, 3]
        u_target = (demand + leak).float()

        attr, adj = pyg_to_ggnn_inputs(data)

        test_attr_buf.append(attr)
        test_adj_buf.append(adj)

        if len(test_attr_buf) > SEQ_LEN:
            test_attr_buf.pop(0)
            test_adj_buf.pop(0)

        if len(test_attr_buf) < SEQ_LEN:
            continue

        with torch.no_grad():
            u_pred = model(test_attr_buf, test_adj_buf)
            
        
        anomaly_time_series.append(u_pred.cpu().numpy())


        #print(f"\nSTEP {step}")
        #print(f"{'Node':<6} {'u_pred':<12} {'u_target':<12} {'diff':<12}")
        for i in range(len(u_pred)):
            p = float(u_pred[i])
            t = float(u_target[i])
            #print(f"{i:<6} {p:<12.5f} {t:<12.5f} {p-t:<12.5f}")

    A = np.array(anomaly_time_series)   # shape: [T, N]

    # somma nel tempo
    susp = A.sum(axis=0)                # shape [N]

    # ranking nodi
    ranking = np.argsort(-susp)

    print("\n=== RANKING NODI SOSPETTI ===")
    for idx in ranking:
        print(f"Nodo {idx:>3}  | S_u = {susp[idx]:.5f}")



if __name__ == "__main__":
    run_GGNN(r"C:\\Users\\nephr\\Desktop\\Uni-Nuova\\Tesi\\Networks-found\\Jilin_copy.inp")