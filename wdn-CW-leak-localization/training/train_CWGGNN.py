import os
import torch
import torch.nn as nn
import networkx as nx
import random

from models.GGNN import GGNNModel
from models.CW_GGNN import TopoCycleResidualNodeAlpha, GGNNWithTopoAlpha
from core.WNTREnv_setup import WNTREnv, build_adj_matrix, sample_binary_mask
from core.face_selection import compute_global_face_scores, reduce_B2_by_scores
from utility.utils_cellcomplex import func_gen_B2_lu


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRAINING
# ============================================================

def train_CWGGNN(inp_path, save_path, EPOCHS=100, num_steps=50, LR=1e-1, LEAK_AREA=0.1, HIDDEN_SIZE=132, PROPAG_STEPS=6, MAX_CYCLE_LENGTH=8, use_pressure=True, use_flow=False, pressure_keep_frac=1.0, flow_keep_frac=1.0, reduce_B2=True, drop_frac=0.5):

    assert use_pressure or use_flow

    # ====== ENV & GRAPH ======
    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)

    cols = list(node2idx.keys())
    N = len(cols)

    # ====== BUILD TOPOLOGY ======
    G = nx.Graph()
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        u = pipe.start_node_name
        v = pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    B1, B2, _ = func_gen_B2_lu(G, MAX_CYCLE_LENGTH)

    print(B2.shape)

    if reduce_B2:
        scores = compute_global_face_scores(inp_path, B1, B2)
        B2, keep_idx = reduce_B2_by_scores(B2, scores, drop_frac=drop_frac)

    print(B2.shape)

    # metti su device
    B1 = torch.tensor(B1, dtype=torch.float32, device=DEVICE)  # [N,E]
    B2 = torch.tensor(B2, dtype=torch.float32, device=DEVICE)  # [E,F]

    # ====== FEATURE SIZE ======
    attr_size = 0
    if use_pressure:
        attr_size += 2
    if use_flow:
        attr_size += 2

    # ====== MODEL ======
    ggnn = GGNNModel(
        attr_size=attr_size,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    ).to(DEVICE)

    topo_node = TopoCycleResidualNodeAlpha(
        B1_np=B1,
        B2_np=B2,
        hidden_dim=HIDDEN_SIZE,
        alpha_max=0.1,        # prova 0.05 / 0.1 / 0.2
    ).to(DEVICE)

    model = GGNNWithTopoAlpha(ggnn, topo_node).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING CW-GGNN ===")
    print(f"attr_size={attr_size}")

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
            e_mask = sample_binary_mask(E, flow_keep_frac)

        df_demand   = results.node["demand"]
        df_leak     = results.node.get("leak_demand", None)

        for t in range(env.leak_start_step, num_steps):

            features = []

            # ---------- PRESSURE ----------
            if use_pressure:
                p = torch.tensor(df_pressure.iloc[t][cols].values, dtype=torch.float32, device=DEVICE)
                p_obs = p * p_mask
                features.append(p_obs)
                features.append(p_mask)

            # ---------- FLOW ----------
            if use_flow:
                q = torch.zeros(E, device=DEVICE)
                row = df_flow.iloc[t]

                for e_idx, (u, v) in enumerate(edge_list):
                    pipe_name = env.wn.get_link_name(u, v)
                    q[e_idx] = float(row[pipe_name])

                q_obs = q * e_mask
                flow_balance = B1 @ q_obs
                flow_mask_node = ((B1.abs() @ e_mask) > 0).float()

                deg = B1.abs().sum(dim=1) + 1e-6
                flow_balance = flow_balance / deg
                flow_balance = flow_balance - flow_balance.mean()

                features.append(flow_balance)
                features.append(flow_mask_node)

            attr = torch.stack(features, dim=-1).unsqueeze(0)

            # ---------- TARGET ----------
            demand = df_demand.iloc[t][cols].values
            leak = df_leak.iloc[t][cols].values if df_leak is not None else 0.0
            target = torch.tensor(demand + leak, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            out = model(attr, adj_matrix)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            #print(f"Loss = {loss.item():.6f}")


    # ====== SAVE TRAINED MODEL ======

    os.makedirs("saved_models", exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "attr_size": attr_size,
        "use_pressure": use_pressure,
        "use_flow": use_flow,
        "hidden_size": HIDDEN_SIZE,
        "propag_steps": PROPAG_STEPS,
        "max_cycle_length": MAX_CYCLE_LENGTH,
    }, save_path,)

    print(f"\n[OK] CW-GGNN salvata in {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"/home/zagaria/Tesi/Tesi/Networks-found-final/GRID_BSD.inp"
    save_path = "saved_models/cwggnn_PF_flowbalance.pt"

    # ====== OPZIONI ======

    use_pressure = True
    use_flow = False
    pressure_keep_frac = 1.0
    flow_keep_frac = 1.0

    reduce_B2 = True
    drop_frac = 0.4

    # ====== IPERPARAMETRI ======
    NUM_STEPS = 50
    EPOCHS = 100
    LR = 1e-2
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 6
    MAX_CYCLE_LENGTH = 8


    train_CWGGNN(inp_path=inp_path, save_path=save_path, EPOCHS=EPOCHS, num_steps=NUM_STEPS, LR=LR, LEAK_AREA=LEAK_AREA, HIDDEN_SIZE=HIDDEN_SIZE, PROPAG_STEPS=PROPAG_STEPS, MAX_CYCLE_LENGTH=MAX_CYCLE_LENGTH, use_pressure=use_pressure, use_flow=use_flow, pressure_keep_frac=pressure_keep_frac, flow_keep_frac=flow_keep_frac, reduce_B2=reduce_B2, drop_frac=drop_frac)
