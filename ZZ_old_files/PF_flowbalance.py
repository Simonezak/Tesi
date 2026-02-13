import os
import torch
import torch.nn as nn
import networkx as nx
import random

from models.GGNN import GGNNModel
from core.WNTREnv_setup import WNTREnv, build_adj_matrix, build_edge2pipe_mapping
from utility.utils_cellcomplex import func_gen_B2_lu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_CWGGNN(
    inp_path,
    EPOCHS=100,
    num_steps=50,
    LR=1e-1,
    LEAK_AREA=0.1,
    HIDDEN_SIZE=132,
    PROPAG_STEPS=6,
    TOPO_MLP_HIDDEN=16,
    MAX_CYCLE_LENGTH=8,
    # ordine delle feature = ordine dei canali in X
    feature_names=("p_obs", "p_mask", "flow_balance", "flow_mask_node"),
    pressure_keep_frac=1.0,
    flow_keep_frac=0.3,
):
    # ================= ENV & GRAPH =================
    env = WNTREnv(inp_path, num_steps=num_steps)
    adj_matrix, node2idx, _ = build_adj_matrix(env.wn)
    adj_matrix = adj_matrix.to(DEVICE)
    cols = list(node2idx.keys())
    N = len(cols)

    # ================= TOPOLOGY & INCIDENCE =================
    edge2pipe = build_edge2pipe_mapping(env.wn, node2idx)

    # costruisci G (ordine di inserimento deterministico = ordine pipe_name_list)
    G = nx.Graph()
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        u = pipe.start_node_name
        v = pipe.end_node_name
        if u in node2idx and v in node2idx:
            G.add_edge(node2idx[u], node2idx[v])

    # B1/B2 (edge order = ordine G.edges())
    B1_np, B2_np, _ = func_gen_B2_lu(G, MAX_CYCLE_LENGTH)

    # metti su device
    B1 = torch.tensor(B1_np, dtype=torch.float32, device=DEVICE)  # [N,E]
    B2 = torch.tensor(B2_np, dtype=torch.float32, device=DEVICE)  # [E,F]

    # IMPORTANT: usa lo STESSO ordine archi usato per B1
    edge_list = list(G.edges())  # [(u,v), ...] in insertion order
    E = len(edge_list)

    # ================= INPUT CONFIG =================
    def sample_binary_mask(n, keep_frac):
        if keep_frac >= 1.0:
            return torch.ones(n, device=DEVICE)
        if keep_frac <= 0.0:
            return torch.zeros(n, device=DEVICE)
        k = max(1, int(round(keep_frac * n)))
        idx = torch.randperm(n, device=DEVICE)[:k]
        m = torch.zeros(n, device=DEVICE)
        m[idx] = 1.0
        return m

    def get_pipe_name(u, v):
        # robusto: prova (u,v), (v,u), e tupla ordinata
        if (u, v) in edge2pipe:
            return edge2pipe[(u, v)]
        if (v, u) in edge2pipe:
            return edge2pipe[(v, u)]
        a, b = (u, v) if u <= v else (v, u)
        return edge2pipe.get((a, b), None)

    def build_X(t, df_pressure, df_flow):
        # masks
        p_mask = sample_binary_mask(N, pressure_keep_frac)
        e_mask = sample_binary_mask(E, flow_keep_frac)

        # ---- pressure ----
        pressure = torch.tensor(df_pressure.iloc[t][cols].values, dtype=torch.float32, device=DEVICE)
        p_obs = pressure * p_mask

        # ---- flow ----
        q = torch.zeros(E, device=DEVICE)
        row = df_flow.iloc[t]
        for e_idx, (u, v) in enumerate(edge_list):
            pn = get_pipe_name(u, v)
            q[e_idx] = 0.0 if pn is None else float(row[pn])

        q_obs = q * e_mask

        # nodal balance (signed) e mask nodale
        flow_balance = B1 @ q_obs                       # [N]
        flow_mask_node = ((B1.abs() @ e_mask) > 0).float()


        #deg = B1.abs().sum(dim=1) + 1e-6
        #flow_balance = flow_balance / deg
        #flow_balance = flow_balance - flow_balance.mean()

        # ---- feature registry ----
        feat_map = {
            "p_obs": p_obs,
            "p_mask": p_mask,
            "flow_balance": flow_balance,
            "flow_mask_node": flow_mask_node
        }

        feats = [feat_map[name] for name in feature_names]
        X = torch.stack(feats, dim=-1).unsqueeze(0)     # [1,N,F]
        return X

    # ================= MODEL =================
    attr_size = len(feature_names)

    ggnn = GGNNModel(
        attr_size=attr_size,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS,
    ).to(DEVICE)

    from models.CW_GGNN import TopoCycleResidualNodeAlpha, GGNNWithTopoAlpha

    topo_node = TopoCycleResidualNodeAlpha(
        B1_np=B1,          # passa tensor su device (supportato dalla patch)
        B2_np=B2,
        hidden_dim=HIDDEN_SIZE,
        alpha_max=0.1,
    ).to(DEVICE)

    model = GGNNWithTopoAlpha(ggnn, topo_node).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n=== TRAINING CW-GGNN (MULTI-INPUT) ===")
    print(f"feature_names={feature_names}")
    print(f"attr_size={attr_size}\n")

    # ================= TRAIN LOOP =================
    for epoch in range(EPOCHS):
        print(f"\n--- Episodio {epoch+1}/{EPOCHS}")

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
        df_flow     = results.link["flowrate"]

        for t in range(env.leak_start_step, num_steps):

            X = build_X(t, df_pressure, df_flow)  # [1,N,F]

            demand = df_demand.iloc[t][cols].values
            leak = df_leak.iloc[t][cols].values if df_leak is not None else 0.0
            target = torch.tensor(demand + leak, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            out = model(X, adj_matrix)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            #print(f"Loss = {loss.item():.6f}")

    # ================= SAVE =================
    os.makedirs("saved_models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attr_size": attr_size,
            "feature_names": feature_names,
            "hidden_size": HIDDEN_SIZE,
            "propag_steps": PROPAG_STEPS,
            "max_cycle_length": MAX_CYCLE_LENGTH,
        },
        "saved_models/cwggnn_branched_mixed_1p03f.pt",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inp_path = r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found-FINAL\20x20_branched_BSD.inp"

    # IPERPARAMETRI
    use_pressure = True
    use_flow     = True

    # IPERPARAMETRI
    EPOCHS = 100
    NUM_STEPS = 50
    LR = 1e-1
    LEAK_AREA = 0.1

    HIDDEN_SIZE = 132
    PROPAG_STEPS = 7
    TOPO_MLP_HIDDEN = 132
    MAX_CYCLE_LENGTH = 8

    train_CWGGNN(
        inp_path=inp_path,
        EPOCHS=EPOCHS,
        num_steps=NUM_STEPS,
        LR=LR,
        LEAK_AREA=LEAK_AREA,
        HIDDEN_SIZE=HIDDEN_SIZE,
        PROPAG_STEPS=PROPAG_STEPS,
        TOPO_MLP_HIDDEN=TOPO_MLP_HIDDEN,
        MAX_CYCLE_LENGTH=MAX_CYCLE_LENGTH,
    )
