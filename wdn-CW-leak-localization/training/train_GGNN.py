
import os
import torch
import torch.nn as nn
import random

from models.GGNN import GGNNModel
from core.WNTREnv_setup import WNTREnv, build_adj_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_GGNN(inp_path, EPOCHS = 50, num_steps = 50, LR = 1e-2, LEAK_AREA = 0.1, HIDDEN_SIZE = 132, PROPAG_STEPS = 7):

    # ====== ENV & GRAPH ======
    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)
    cols = list(node2idx.keys())

    # ====== MODEL ======
    model = GGNNModel(
        attr_size=1,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING GGNN ===")

    # ====== TRAIN LOOP ======
    for epoch in range(EPOCHS):

        print(f"\n--- Episodio {epoch+1}/{EPOCHS}")
        
        #num_leaks = random.randint(1, 2)
        num_leaks = 2
        env.reset(num_leaks)
        sim = env.sim

        for step in range(num_steps):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=LEAK_AREA)

            sim.step_sim()


        results = sim.get_results()

        df_pressure = results.node["pressure"]
        df_demand   = results.node["demand"]
        df_leak     = results.node.get("leak_demand", None)

        for t in range(env.leak_start_step, num_steps):

            # ---------- INPUT ----------
            p = torch.tensor(
                df_pressure.iloc[t][cols].values,
                dtype=torch.float32
            ).to(DEVICE)

            attr = p.unsqueeze(0).unsqueeze(-1)

            # ---------- TARGET ----------
            demand = df_demand.iloc[t][cols].values
            leak = df_leak.iloc[t][cols].values

            target = torch.tensor(
                demand + leak,
                dtype=torch.float32
            ).to(DEVICE)

            # ---------- OPTIM ----------
            optimizer.zero_grad()
            out = model(attr, adj_matrix)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")

    # ====== SAVE TRAINED MODEL ======

    os.makedirs("saved_models", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "attr_size": 1,
        "hidden_size": HIDDEN_SIZE,
        "propag_steps": PROPAG_STEPS
    }, "saved_models/ggnn_model.pt")

    print("\n[OK] GGNN salvata in saved_models/ggnn_model.pt")



if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp"

    # IPERPARAMETRI
    EPOCHS = 50
    num_steps = 50
    LR = 1e-2
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 7

    train_GGNN(inp_path, EPOCHS, num_steps, LR, LEAK_AREA, HIDDEN_SIZE, PROPAG_STEPS)
