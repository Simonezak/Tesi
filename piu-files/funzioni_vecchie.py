def run_GNN_experiment(inp_path):

    from torch_geometric.utils import to_dense_adj
    from GNN_LD import GNNLeakDetector
    import torch.nn as nn

    # ---------------------------
    # 1️⃣ Setup ambiente
    # ---------------------------

    print("\n=== 💧 Training GNN on a single LEAK scenario ===")
    env = WNTREnv(inp_path, max_steps=5, hydraulic_timestep=3600)
    env.reset(with_leak=True)
    wn = env.wn
    sim = env.sim
    leak_node = env.leak_node_name
    print(f"[INFO] Leak at node: {leak_node}")

    graphs, labels = [], []

    # 🔹 Step 1: Simulazione su più timestep
    for step in range(env.max_steps):
        sim.step_sim()
        results = sim.get_results()
        data, node2idx, idx2node, edge2idx, idx2edge = build_pyg_from_wntr(wn, results, -1)

        # label = 1 per il nodo con perdita
        y = torch.zeros(data.num_nodes, 1)
        if leak_node in node2idx:
            y[node2idx[leak_node]] = 1.0
        else:
            print(f"[WARN] Leak node {leak_node} not in current graph.")

        graphs.append(data)
        labels.append(y)

    # 🔹 Step 2: Costruisci modello
    model = GNNLeakDetector(node_in_dim=graphs[0].x.shape[1], hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print("\n[TRAINING] Starting...")
    for epoch in range(50):
        total_loss = 0.0
        for data, y in zip(graphs, labels):
            model.train()
            optimizer.zero_grad()

            preds = model(data)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(graphs):.4f}")

    # 🔹 Step 3: Testa sull’ultimo timestep
    model.eval()
    with torch.no_grad():
        preds = model(graphs[-1])
        preds_bin = (preds > 0.5).float()

    print("\n=== Risultati finali ===")
    print(f"Leak reale: {leak_node}")

    probs = preds.squeeze().detach().cpu()       # Conversione del tensore in array leggibile
    topk = torch.topk(probs, k=3)                # Prende le 3 probabilità più alte

    for rank, (idx, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        node_name = idx2node[idx]                # Converte indice interno → nome nodo WNTR
        print(f" {rank}. Nodo {node_name} → prob = {val:.4f}")

    G, coords = build_nx_graph_from_wntr(wn, results)
    plot_leak_probability(G, coords, preds, node2idx[env.leak_node_name])


def build_pyg_from_wntr(
    wn,
    results,
    timestep_index: int,
    cfg: GraphFeatureConfig = GraphFeatureConfig(),
):
    import numpy as np
    import pandas as pd
    from torch_geometric.data import Data
    from wntr.network.elements import Junction, LinkStatus

    # ---- nodi ----
    node_names: List[str] = [name for name, _ in wn.nodes()]  # includi TUTTI i nodi
    node2idx = {name: i for i, name in enumerate(node_names)}
    idx2node = {i: name for name, i in node2idx.items()}

    # Lettura attributi di ogni nodo
    elev, demand, pressure, leak_dem = [], [], [], []
    df_demand: Optional[pd.DataFrame] = results.node.get("demand", None)
    df_pressure: Optional[pd.DataFrame] = results.node.get("pressure", None)
    df_leak: Optional[pd.DataFrame] = results.node.get("leak_demand", None)

    for name in node_names:
        n = wn.get_node(name)
        elev.append(float(getattr(n, "elevation", 0.0)))
        demand.append(safe_get(df_demand, timestep_index, name))
        pressure.append(safe_get(df_pressure, timestep_index, name))
        leak_dem.append(safe_get(df_leak, timestep_index, name))
        x = np.stack([elev, demand, pressure, leak_dem], axis=1)
        x_torch = torch.tensor(x, dtype=torch.float32)

    # ---- archi (pipes) ----
    edge_index_list: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    edge_names: List[str] = []

    forward_edge_idx_for_pipe: List[int] = []  # mapping pipe_id -> edge_index (direzione forward)
    pipe_names: List[str] = []

    df_flow = results.link.get("flowrate", None)

    lengths, diameters, flows, starts, ends, statuses = [], [], [], [], [], []


    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        u_name, v_name = pipe.start_node_name, pipe.end_node_name
        if u_name not in node2idx or v_name not in node2idx:
            continue

        # controlla stato (solo se pipe aperta)
        if pipe.status != LinkStatus.Open:
            continue
        status = 1.0 if pipe.status == LinkStatus.Open else 0.0

        u, v = node2idx[u_name], node2idx[v_name]
        length = float(getattr(pipe, "length", 0.0))
        diameter = float(getattr(pipe, "diameter", 0.0))
        flow = safe_get(df_flow, timestep_index, pipe_name)

        # aggiungi forward edge
        forward_idx = len(edge_index_list)
        edge_index_list.append((u, v))
        edge_attrs.append([length, diameter, flow])
        edge_names.append(pipe_name)
        forward_edge_idx_for_pipe.append(forward_idx)
        pipe_names.append(pipe_name)

        lengths.append(length)
        diameters.append(diameter)
        flows.append(flow)
        starts.append(u_name)
        ends.append(v_name)
        statuses.append(status)

        # se grafo non orientato, aggiungi anche reverse
        if cfg.undirected:
            edge_index_list.append((v, u))
            edge_attrs.append([length, diameter, flow])
            edge_names.append(f"{pipe_name}__rev")

    #all_pipes = wn.pipe_name_list   
    #pipe_edge_idx = []
    #pipe_open_mask = []

    """
    for pipe_name in all_pipes:
        pipe = wn.get_link(pipe_name)
        # Trova se il tubo compare nel grafo (solo se era aperto)
        if pipe_name in edge_names:
            idx = edge_names.index(pipe_name)
            pipe_edge_idx.append(idx)
            pipe_open_mask.append(1.0)  # tubo aperto e presente nel grafo
        else:
            pipe_edge_idx.append(-1)    # tubo chiuso, nessun arco nel grafo
            pipe_open_mask.append(0.0)  # 0 = chiuso, ma azione ancora possibile
    """
            
    # ---- costruisci Data PyG ----
    edge_index = torch.tensor(np.array(edge_index_list, dtype=np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32), dtype=torch.float32)

    data = Data(x=x_torch, edge_index=edge_index, edge_attr=edge_attr)
    #data.num_nodes = x.shape[0]

    # campi extra
    #data.pipe_edge_idx = torch.tensor(pipe_edge_idx, dtype=torch.long)
    #data.pipe_open_mask = torch.tensor(pipe_open_mask, dtype=torch.float32)
    #data.pipe_names = all_pipes
    #data.num_pipes = len(forward_edge_idx_for_pipe)

    edge2idx = {name: i for i, name in enumerate(edge_names)}
    idx2edge = {i: name for name, i in edge2idx.items()}

    return data, node2idx, idx2node, edge2idx, idx2edge



def step(self, action_index):

    self.global_step += 1

    if action_index < 2 * self.num_pipes:
        pipe_id = action_index // 2
        act = action_index % 2  # 0=close, 1=open
        pipe_name = self.data.pipe_names[pipe_id]

        if act == 0:
            close_pipe(self.sim, pipe_name)
        else:
            open_pipe(self.sim, pipe_name)

    elif action_index == 2 * self.num_pipes:
        noop(self.sim)

    #elif action_index == 2 * self.num_pipes + 1:
    #    close_all_pipes(self.sim)

    else:
        raise ValueError(f"Azione fuori range: {action_index}")

    # Avanza la simulazione di un passo
    self.sim.step_sim()
    self.results = self.sim.get_results()
    #print(self.results.node["pressure"].iloc[-1])

    next_state, *_ = build_pyg_from_wntr(self.wn, self.results)

    # Reward: pressione media vicina a 50
    pressures = next_state.x[:, 2].mean().item()
    reward = -abs(pressures - 50.0)

    self.current_step += 1
    print(self.current_step)
    done = self.current_step >= self.max_steps or self.sim.is_terminated()
    return next_state, reward, done, {}



def run_GNN_topo_comparison(inp_path):
    """
    - genera i grafi PyG agli step 1..5
    - allena entrambi i modelli su tutti gli step
    - valuta e plotta a step 1 e step 5:
        * plot_leak_probability (entrambi)
        * plot_node_demand / plot_edge_flowrate / plot_cell_complex_flux (contesto)
    """

    # Parametri interni default
    max_steps   = 5
    epochs      = 50
    lr          = 1e-3
    hidden_dim  = 64
    topo_proj   = 32
    dropout     = 0.2

    # 0) Setup ambiente + leak
    print("\n=== 💧 Confronto GCN semplice vs GCN+TopoLayer ===")
    env = WNTREnv(inp_path, max_steps=max_steps, hydraulic_timestep=3600)
    env.reset(with_leak=True)
    wn, sim = env.wn, env.sim
    leak_name = env.leak_node_name
    print(f"[INFO] Leak at node: {leak_name}")

    # ---------------------------
    # 1) Raccolta dataset (1..max_steps)
    # ---------------------------
    graphs = []      # Data PyG
    labels = []      # y per-nodo
    aux    = []      # (G, coords, results, B1,B2,f,f_polygons, node2idx, idx2node)

    for step in range(1, max_steps + 1):
        print(f"  > Sim step {step}/{max_steps}")
        sim.step_sim()
        results = sim.get_results()

        data, node2idx, idx2node, edge2idx, idx2edge = build_pyg_from_wntr(wn, results)

        # Label per-nodo
        y = torch.zeros(data.num_nodes, 1, dtype=torch.float32)
        if leak_name in node2idx:
            y[node2idx[leak_name]] = 1.0
        else:
            print(f"[WARN] Leak node {leak_name} non presente nel grafo PyG allo step {step}")

        graphs.append(data)
        labels.append(y)

        # Oggetti per i plot
        G, coords = build_nx_graph_from_wntr(wn, results)
        B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)
        f = construct_matrix_f(wn, results)
        f_polygons = compute_polygon_flux(f, B2, False)

        aux.append((G, coords, results, B1, B2, f, f_polygons, node2idx, idx2node))

    # ---------------------------
    # 2) Modelli
    # ---------------------------
    sample = graphs[0]
    node_in_dim = sample.x.shape[1]
    topo_in_dim = getattr(sample, "topo", None).shape[1] if hasattr(sample, "topo") and sample.topo is not None else 0

    model_plain = GNNLeakDetector(node_in_dim=node_in_dim, hidden_dim=hidden_dim, dropout=dropout)
    model_topo  = GNNLeakDetectorTopo(node_in_dim=node_in_dim, topo_in_dim=topo_in_dim,
                                      hidden_dim=hidden_dim, topo_proj_dim=topo_proj, dropout=dropout)

    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=lr)
    opt_topo  = torch.optim.Adam(model_topo.parameters(),  lr=lr)
    loss_fn = nn.BCELoss()

    # ---------------------------
    # 3) Training (stessa procedura per entrambi)
    # ---------------------------

    print("\n[TRAIN] GCN semplice")
    train_model(model_plain, opt_plain, graphs, labels, epochs=epochs, name="GCN")

    print("\n[TRAIN] GCN + TopoLayer")
    train_model(model_topo,  opt_topo,  graphs, labels, epochs=epochs, name="GCN+Topo")

    # ---------------------------
    # 4) Valutazione & Plot a step 1 e 5
    # ---------------------------

    return model_plain, model_topo, graphs