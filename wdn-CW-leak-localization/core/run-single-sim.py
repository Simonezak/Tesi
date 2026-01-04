def run_GGNN(inp_path):
    """
    Prova del modello GGNN di Leveraging, che
        1) usa solo la pressione dei nodi e matrice di adiacenza per predire leak
        2) non ha topological layer
    """

    num_episodes = 200
    max_steps    = 50
    lr           = 1e-2
    epochs       = 500
    area = 0.1
    HIDDEN_SIZE = 132
    PROPAG_STEPS = 7
    WINDOW_SIZE = 4 
    

    all_snapshots_with_leak = []
    rf_training_data = []

    env = WNTREnv(inp_path, max_steps=max_steps)

    # costruisci adiacency matrix e indici UNA VOLTA all'inizio dato che non cambiano
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(env.wn)

    # Inizia Simulazione WNTR
    for ep in range(num_episodes):
        print(f"\n--- Episodio {ep+1}/{num_episodes}")
        
        n_leaks = np.random.randint(1, 3)
        env.reset(num_leaks=n_leaks)
        sim = env.sim

        episode_feature_vectors = []

        for step in range(max_steps):

            if step == env.leak_start_step:
                for leak_node in env.leak_node_names:  
                    env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

            sim.step_sim()


            """
            for leak_node in env.leak_node_names:
                leak_idx = node2idx[leak_node]
                leak_val = data.x[leak_idx, 3].item()

                print(f"Step {step}: leak_demand[{leak_node}] = {leak_val:.6f}")
            """       

        results = sim.get_results()

        df_pressure = results.node["pressure"]       # shape [T, N]
        df_demand   = results.node["demand"]         # shape [T, N]
        df_leak     = results.node.get("leak_demand", None)

        # Aggiungi ogni riga del dataframe
        cols = list(node2idx.keys())
        episode_feature_vectors = df_pressure[cols].to_numpy(dtype=np.float32).tolist()

        #if ep == 1:
            #sim.plot_results("node", "demand")
            #sim.plot_network_over_time("demand", "flowrate")
            #sim.plot_network()

        rf_training_data.append({
            "feature_vector": episode_feature_vectors,
            "leak_start": env.leak_start_step
        })


    cols = list(node2idx.keys())

    P = df_pressure[cols].to_numpy(dtype=np.float32)   # [T, N]
    D = df_demand[cols].to_numpy(dtype=np.float32)     # [T, N]

    if df_leak is None:
        L = np.zeros_like(D)
    else:
        L = df_leak[cols].to_numpy(dtype=np.float32)

        
    T, N = P.shape

    for t in range(WINDOW_SIZE - 1, T):

        # finestra pressione [W, N]
        window = P[t - WINDOW_SIZE + 1 : t + 1]     # [W, N]

        # attr_matrix [1, N, W]
        attr_matrix = torch.tensor(
            window.T, dtype=torch.float32
        ).unsqueeze(0)

        # target solo dopo leak onset
        if t < env.leak_start_step:
            continue

        u = D[t] + L[t]                              # [N]
        y = torch.tensor(u, dtype=torch.float32).view(-1, 1)

        all_snapshots_with_leak.append({
            "attr": attr_matrix,
            "adj":  adj_matrix,
            "y":    y
        })





    # ============================================================
    #            TRAIN RANDOM FOREST LEAK-ONSET
    # ============================================================

    print("\n=== TRAINING RANDOM FOREST ===")
    rf = RandomForestLeakOnsetDetector()
    rf.fit(rf_training_data)

    # ============================================================
    #                       TRAIN GGNN
    # ============================================================

    model = GGNNModel(
        attr_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        propag_steps=PROPAG_STEPS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    print("\n=== TRAINING GGNN ===")

    for epoch in range(epochs):
        
        model.train()

        sample = np.random.choice(all_snapshots_with_leak)

        attr = sample["attr"]
        adj  = sample["adj"]
        y    = sample["y"]  # [N,1]

        # target ora è [1,N]
        target = y.squeeze().float().unsqueeze(0)

        optimizer.zero_grad()
        out = model(attr, adj) # output [1,N]

        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss={loss.item():.8f}")



    print("\n\n=== TEST PHASE ===")

    test_env = WNTREnv(inp_path, max_steps=max_steps)
    adj_matrix, node2idx, idx2node = build_static_graph_from_wntr(test_env.wn)
    n_leaks = np.random.randint(1, 3)
    test_env.reset(num_leaks=n_leaks)
    sim = test_env.sim

    test_snapshots = []
    test_pressure_window = []


    # --------------------
    # 1) LEAK ONSET DETECTION (RandomForest)
    # --------------------

    print("\n--- Leak detection (Random Forest) ---")

    onset_scores = []

    for step in range(max_steps):

        # attiva leak nel momento corretto
        if step == test_env.leak_start_step:
            for leak_node in test_env.leak_node_names:
                test_env.sim.start_leak(leak_node, leak_area=area, leak_discharge_coefficient=0.75)

        sim.step_sim()


    results = sim.get_results()

    df_pressure = results.node["pressure"]
    df_demand   = results.node["demand"]
    df_leak     = results.node.get("leak_demand", None)

    cols = list(node2idx.keys())

    for t in range(len(df_pressure)):

        pressures = df_pressure.loc[:, cols].iloc[t].to_numpy(dtype=np.float32)
        demand    = df_demand.loc[:, cols].iloc[t].to_numpy(dtype=np.float32)
        leak = df_leak.loc[:, cols].iloc[t].to_numpy(dtype=np.float32)

        # salvalo in lista
        test_snapshots.append({
            "pressures": pressures,
            "demand":    demand,
            "leak":      leak
        })

        prob = rf.predict(pressures)
        onset_scores.append(prob)

    predicted_onset = int(np.argmax(onset_scores))
    print(f"\n Inizio leak stimato allo step: {predicted_onset}")

    anomaly_time_series = []

    # --------------------
    # 2) LEAK LOCALIZATION (GGNN) - PER OGNI STEP DOPO ONSET
    # --------------------

    TOTAL_STEPS = len(test_snapshots)

    for snap in test_snapshots[predicted_onset:]:

        current_pressures = torch.tensor(snap["pressures"], dtype=torch.float32)  # [N]
        test_pressure_window.append(current_pressures)

        if len(test_pressure_window) > WINDOW_SIZE:
            test_pressure_window.pop(0)
        if len(test_pressure_window) < WINDOW_SIZE:
            continue

        attr_matrix = build_attr_from_pressure_window(test_pressure_window)  # [1,N,W]
        with torch.no_grad():
            u_pred = model(attr_matrix, adj_matrix).view(-1)

        anomaly_time_series.append(u_pred.cpu().numpy())
        """
        if step >= TOTAL_STEPS - 1:
            # target
            df_demand = results.node["demand"]
            df_leak = results.node.get("leak_demand", None)

            # Estrai demand e leak come numpy
            demand = np.array([df_demand.loc[:, name].values[-1] for name in node2idx.keys()], dtype=np.float32)
            leak = np.array([df_leak.loc[:, name].values[-1] for name in node2idx.keys()], dtype=np.float32) if df_leak is not None else np.zeros_like(demand)


            # Converti in tensori PyTorch
            demand = torch.tensor(demand, dtype=torch.float32)
            leak = torch.tensor(leak, dtype=torch.float32)

            # Calcola u_target
            u_target = (demand + leak).view(-1)

            print(f"{'Nodo':<8} {'u_pred':<12} {'demand':<12} {'leak':<12} {'u_target':<12} {'diff':<12}")
            print("-" * 70)

            for i in range(len(u_pred)):
                node_name = idx2node[i]

                p = float(u_pred[i])
                d = float(demand[i])
                l = float(leak[i])
                t = float(u_target[i])

                print(
                    f"{node_name:<8} "
                    f"{p:<12.5f} "
                    f"{d:<12.5f} "
                    f"{l:<12.5f} "
                    f"{t:<12.5f} "
                    f"{(p - t):<12.5f}"
                )
            
            print("\n\n")
            """

    print("\n\n=== RANKING NODI PER ANOMALIA CUMULATA (basato su u_pred) ===")

    A = np.array(anomaly_time_series)   # shape [T, N]
    T, N = A.shape

    # 🔹 somma temporale delle anomalie per nodo
    score = A.sum(axis=0)               # [N]

    # ranking decrescente
    ranking = np.argsort(-score)

    print(f"\n{'Nodo':<10} {'score (Σ u_pred)':<20}")
    print("-" * 35)

    for idx in ranking:
        print(f"{idx2node[idx]:<10} {score[idx]:<20.8f}")

    print("\nNodi leak reali:", test_env.leak_node_names)


if __name__ == "__main__":
    run_GGNN(inp_path=r"/home/zagaria/Tesi/Tesi/Networks-found/20x20_branched.inp")