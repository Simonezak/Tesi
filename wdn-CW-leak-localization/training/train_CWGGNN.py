import os
import torch
import torch.nn as nn
import networkx as nx
import random

from models.GGNN import GGNNModel
from core.WNTREnv_setup import WNTREnv, build_adj_matrix
from utility.utils_cellcomplex import func_gen_B2_lu


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRAINING
# ============================================================

def train_CWGGNN(inp_path, EPOCHS=100, num_steps=50, LR=1e-1, LEAK_AREA=0.1, HIDDEN_SIZE=132, PROPAG_STEPS=6, TOPO_MLP_HIDDEN=16, MAX_CYCLE_LENGTH=8):

    # ====== ENV & GRAPH ======
    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)
    cols = list(node2idx.keys())

    # ====== BUILD TOPOLOGY ======
    G = nx.Graph()
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        u = pipe.start_node_name
        v = pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    B1, B2, _ = func_gen_B2_lu(G, MAX_CYCLE_LENGTH)
    L1 = B1.T @ B1 + B2 @ B2.T

    # ====== MODEL ======
    ggnn = GGNNModel(
        attr_size=1,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    ).to(DEVICE)

    from models.CW_GGNN import TopoCycleResidualNodeAlpha, GGNNWithTopoAlpha

    topo_node = TopoCycleResidualNodeAlpha(
        B1_np=B1,
        B2_np=B2,
        hidden_dim=HIDDEN_SIZE,
        alpha_max=0.1,        # prova 0.05 / 0.1 / 0.2
        use_layernorm=True
    ).to(DEVICE)

    model = GGNNWithTopoAlpha(ggnn, topo_node).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING CW-GGNN ===")

    # ====== TRAIN LOOP ======
    for epoch in range(EPOCHS):

        print(f"\n--- Episodio {epoch+1}/{EPOCHS}")
        
        num_leaks = random.randint(2, 3)
        #num_leaks = 2
        env.reset(num_leaks)
        sim = env.sim

        env.leak_start_step = 20

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

            target = torch.tensor(demand + leak,dtype=torch.float32).to(DEVICE)

            # ---------- OPTIM ----------
            optimizer.zero_grad()
            out = model(attr, adj_matrix)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            print(f"Loss = {loss.item():.6f}")

    # ====== SAVE TRAINED MODEL ======

    os.makedirs("saved_models", exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "attr_size": 1,
        "hidden_size": HIDDEN_SIZE,
        "propag_steps": PROPAG_STEPS,
        "topo_mlp_hidden": TOPO_MLP_HIDDEN,
        "max_cycle_length": MAX_CYCLE_LENGTH,
    }, "saved_models/cw_ggnn_Modena.pt") 


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found-final/modena_BSD.inp"

    # ====== IPERPARAMETRI ======
    NUM_STEPS = 50
    EPOCHS = 50
    LR = 2e-1
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6
    TOPO_MLP_HIDDEN = 32
    MAX_CYCLE_LENGTH = 8


    train_CWGGNN(inp_path=inp_path, EPOCHS=EPOCHS, num_steps=NUM_STEPS, LR=LR, LEAK_AREA=LEAK_AREA, HIDDEN_SIZE=HIDDEN_SIZE, PROPAG_STEPS=PROPAG_STEPS, TOPO_MLP_HIDDEN=TOPO_MLP_HIDDEN, MAX_CYCLE_LENGTH=MAX_CYCLE_LENGTH)
