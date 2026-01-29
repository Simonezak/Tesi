
import os
import torch
import torch.nn as nn
import random
import networkx as nx

from models.GGNN import GGNNModel
from core.WNTREnv_setup import WNTREnv, build_adj_matrix, sample_binary_mask, build_edge2pipe_mapping
from utility.utils_cellcomplex import func_gen_B2_lu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_GGNN(inp_path, save_path, use_pressure, use_flow, pressure_keep_frac = 1.0, flow_keep_frac = 1.0, EPOCHS = 50, num_steps = 50, LR = 1e-2, LEAK_AREA = 0.1, HIDDEN_SIZE = 132, PROPAG_STEPS = 7):

    assert use_pressure or use_flow

    # ====== ENV & GRAPH ======
    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)
    
    cols = list(node2idx.keys())
    N = len(cols)

    # -------- attr_size --------
    attr_size = 0
    if use_pressure:
        attr_size += 2
    if use_flow:
        attr_size += 2

        edge2pipe = build_edge2pipe_mapping(env.wn, node2idx)
        edge_list = sorted(edge2pipe.keys())

        G = nx.Graph()
        for pipe_name in env.wn.pipe_name_list:
            pipe = env.wn.get_link(pipe_name)
            u = pipe.start_node_name
            v = pipe.end_node_name
            if u in node2idx and v in node2idx:
                G.add_edge(node2idx[u], node2idx[v])

        B1, B2, _ = func_gen_B2_lu(G, 8)
        B1 = torch.tensor(B1, dtype=torch.float32, device=DEVICE)


    # ====== MODEL ======
    model = GGNNModel(
        attr_size=attr_size,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING GGNN ===")
    print(f"attr_size = {attr_size}\n")

    # ====== TRAIN LOOP ======
    for epoch in range(EPOCHS):

        print(f"\n--- Episodio {epoch+1}/{EPOCHS}")
        
        #num_leaks = random.randint(2, 3)
        num_leaks = 2
        env.reset(num_leaks)
        sim = env.sim

        for step in range(num_steps):

            if step == env.leak_start_step:
                for ln in env.leak_node_names:
                    sim.start_leak(ln, leak_area=LEAK_AREA)

            sim.step_sim()

        results = sim.get_results()

        if use_pressure:
            df_pressure = results.node["pressure"]
            p_mask = sample_binary_mask(N, pressure_keep_frac)

        if use_flow:
            df_flow = results.link["flowrate"]
            e_mask = sample_binary_mask(len(edge_list), flow_keep_frac)

        df_demand   = results.node["demand"]
        df_leak     = results.node.get("leak_demand", None)

        for t in range(env.leak_start_step, num_steps):

            features = []

            if use_pressure:
                pressure = torch.tensor(
                    df_pressure.iloc[t][cols].values,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                p_obs = pressure * p_mask
                features.append(p_obs)
                features.append(p_mask)

            if use_flow:
                q = torch.zeros(len(edge_list), device=DEVICE)
                row = df_flow.iloc[t]
                for e_idx, (u, v) in enumerate(edge_list):
                    pipe_name = edge2pipe[(u, v)]
                    q[e_idx] = float(row[pipe_name])

                q_obs = q * e_mask
                flow_balance = B1 @ q_obs
                flow_mask_node = ((B1.abs() @ e_mask) > 0).float()

                deg = B1.abs().sum(dim=1) + 1e-6
                flow_balance = flow_balance / deg
                flow_balance = flow_balance - flow_balance.mean()

                features.append(flow_balance)
                features.append(flow_mask_node)

            attr = torch.stack(features, dim=-1)

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


        print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")

    # ====== SAVE TRAINED MODEL ======

    os.makedirs("saved_models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": HIDDEN_SIZE,
            "propag_steps": PROPAG_STEPS,
            "attr_size": attr_size,
        },
        save_path,)

    print(f"\n[OK] GGNN salvata in {save_path}")



if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found-final/GRID_BSD.inp"
    save_path = "saved_models/ggnn_PF_flowbalance.pt"

    use_pressure = True
    use_flow     = False

    pressure_keep_frac = 1.0
    flow_keep_frac     = 1.0

    # IPERPARAMETRI
    EPOCHS = 100
    num_steps = 50
    LR = 1e-2
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 7

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    train_GGNN(inp_path, save_path, use_pressure, use_flow, pressure_keep_frac, flow_keep_frac, EPOCHS, num_steps, LR, LEAK_AREA, HIDDEN_SIZE, PROPAG_STEPS)
