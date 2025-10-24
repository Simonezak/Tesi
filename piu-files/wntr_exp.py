import torch
import torch.optim as optim
import random
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

from DQN.gnn import DQNGNN
from DQN.replay_buffer import ReplayBuffer
from DQN.train_utils import seed_torch, train, device
from data_utils.wntr_to_pyg import build_pyg_from_wntr
from actions import open_pipe, close_pipe, close_all_pipes, noop
import matplotlib.pyplot as plt
from main_dyn_topologyknown_01 import func_gen_B2_lu, plot_cell_complex

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, max_steps=5, hydraulic_timestep=3600):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.sim = None
        self.wn = None
        self.results = None
        self.global_step = 0
        self.current_step = 0
        self.done = False

    def reset(self, with_leak=True):
        import numpy as np
        import wntr

        # ===============================
        # Crea rete base
        # ===============================
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)

        # ===============================
        # Aggiungi un leak realistico
        # ===============================

        self.leak_node_name = None

        if with_leak:
            junctions = [name for name, node in self.wn.nodes()
                        if isinstance(node, wntr.network.elements.Junction)]
            leak_node_name = np.random.choice(junctions)
            self.leak_node_name = "J1"
            print(f"[LEAK] Nodo selezionato per la perdita: {leak_node_name}")

            # parametri leak
            D_ref = 0.3
            leak_area_m2 = 5e-1
            leak_k_scale = 1.0
            leak_area_jitter = 0.1
            jitter = 1.0 + leak_area_jitter * float(np.random.uniform(-1.0, 1.0))
            area_eff = leak_area_m2 * (D_ref ** 2) * leak_k_scale * jitter * 5

            # usa il metodo nativo di InteractiveWNTRSimulator
            self.sim.start_leak(leak_node_name, leak_area=float(area_eff), leak_discharge_coefficient=0.75)
            print(f"[LEAK] Aggiunta perdita (area effettiva={area_eff:.2e} m²)")

        else:
            print("[INIT] Nessuna perdita inserita in questo episodio.")

        # ===============================
        # 3️⃣ Inizializza simulatore WNTR
        # ===============================
        self.sim = InteractiveWNTRSimulator(self.wn)
        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        # Esegui un passo iniziale
        self.sim.step_sim()
        self.results = self.sim.get_results()

        df_flow = self.results.link["flowrate"]

        # Prendi l'ultimo timestep simulato (dopo il primo step)
        flow_at_last_timestep = df_flow.iloc[-1] 
        
        flow_values = []
        pipe_names_valid = []
        for pipe_name in self.wn.pipe_name_list:
            pipe = self.wn.get_link(pipe_name)
            start_node = self.wn.get_node(pipe.start_node_name)
            end_node = self.wn.get_node(pipe.end_node_name)
            # includi solo i tubi che collegano junction–junction
            if start_node.__class__.__name__ == "Junction" and end_node.__class__.__name__ == "Junction": 
                # questo è stato fatto perche f quando veniva costruito con pyg includeva anche le reservoir ma nx non le include perche sono inutili al nostro contesto di base lol
                flow_values.append(flow_at_last_timestep[pipe_name])
                pipe_names_valid.append(pipe_name)

        self.f = np.array(flow_values, dtype=float).reshape(-1, 1)
        self.pipe_names_valid = pipe_names_valid  # salva la lista coerente con f

        # f: vettore colonna [Nedge × 1]
        self.f = np.array(flow_values, dtype=float).reshape(-1, 1)

        # ===============================
        # 4️⃣ Costruisci grafo PyG
        # ===============================
        self.data, *_ = build_pyg_from_wntr(self.wn, self.results, -1)
        self.num_pipes = len(self.wn.pipe_name_list)
        self.action_dim = 2 * self.num_pipes  # 0=close, 1=open per ciascun tubo
        self.current_step = 0
        self.done = False

        print("[RESET] Step iniziale completato, grafo creato.")
        return self.data

    
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

        next_state, *_ = build_pyg_from_wntr(self.wn, self.results, -1)

        # Reward: pressione media vicina a 50
        pressures = next_state.x[:, 2].mean().item()
        reward = -abs(pressures - 50.0)

        self.current_step += 1
        print(self.current_step)
        done = self.current_step >= self.max_steps or self.sim.is_terminated()
        return next_state, reward, done, {}



def run_wntr_experiment(inp_path):
    seed = 42
    random.seed(seed)
    seed_torch(seed)

    env = WNTREnv(inp_path, max_steps=5)

    episodes = 1
    learning_rate = 5e-4
    target_update_interval = 5
    train_interval = 1000

    q = DQNGNN().to(device)
    q_target = DQNGNN().to(device)
    q_target.load_state_dict(q.state_dict())

    #memory = ReplayBuffer()
    #optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    #score_list = []

    print("\n=== 💧 Simulazione con LEAK grande ===")
    s = env.reset(with_leak=True)

    wn = env.wn  # modello WNTR caricato

    print("\n🔍 Controllo domande ai nodi:")
    total_demand = 0.0
    zero_demand_nodes = []
    for name, node in wn.nodes():
        if isinstance(node, wntr.network.elements.Junction):
            # ✅ usa base_value invece di base_demand
            base_demand = sum(ts.base_value for ts in node.demand_timeseries_list)
            total_demand += base_demand
            if base_demand <= 0:
                zero_demand_nodes.append(name)
            print(f"  Nodo {name:10s} → domanda base = {base_demand:.6f} m³/s")

    print(f"\n💧 Domanda totale = {total_demand:.6f} m³/s")
    if len(zero_demand_nodes) > 0:
        print(f"⚠️ Nodi senza domanda: {zero_demand_nodes}")
    else:
        print("✅ Tutti i nodi hanno domanda > 0")



    print("\n🔍 Controllo perdita (leak):")
    leak_node_name = getattr(env, "leak_node_name", None)
    if leak_node_name is None:
        print("⚠️ Nessun leak registrato in env.leak_node_name")
    else:
        print(f"  Leak node selezionato: {leak_node_name}")
        leak_node = wn.get_node(leak_node_name)


    # 🔹 Primo step (t = 1h)
    print("\n🕐 Step iniziale (1h)")
    env.sim.step_sim()                 # avanza di 1 ora
    env.results = env.sim.get_results()

    # aggiorna vettore f
    df_flow = env.results.link["flowrate"]

    flow_at_last_timestep = df_flow.iloc[-1]
    flow_values = []
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        start_node = env.wn.get_node(pipe.start_node_name)
        end_node = env.wn.get_node(pipe.end_node_name)
        # includi solo i tubi tra Junction–Junction
        if start_node.__class__.__name__ == "Junction" and end_node.__class__.__name__ == "Junction":
            flow_values.append(flow_at_last_timestep[pipe_name])

    env.f = np.array(flow_values, dtype=float).reshape(-1, 1)

    # Plot iniziale
    vmin, vmax = plot_topological_cycles(
        env.wn, env.data, env.f,
        max_cycle_length=8,
        leak_node_name=env.leak_node_name, step=1
    )
    plt.show()

    # 🔹 Altri 4 step (fino a 5h)
    for step in range(2, 10):
        print(f"🕒 Step {step}/5 (1h)")
        env.sim.step_sim()
        env.results = env.sim.get_results()

    # ✅ Ricalcola f finale (solo tubi J–J)
    df_flow = env.results.link["flowrate"]
    print(df_flow)
    flow_at_last_timestep = df_flow.iloc[-1]
    flow_values = []
    for pipe_name in env.wn.pipe_name_list:
        pipe = env.wn.get_link(pipe_name)
        start_node = env.wn.get_node(pipe.start_node_name)
        end_node = env.wn.get_node(pipe.end_node_name)
        if start_node.__class__.__name__ == "Junction" and end_node.__class__.__name__ == "Junction":
            flow_values.append(flow_at_last_timestep[pipe_name])
    env.f = np.array(flow_values, dtype=float).reshape(-1, 1)
    print(f"[FLOW] f finale (dopo 5h): {env.f.shape}")

    plot_topological_cycles(
        env.wn, env.data, env.f,
        max_cycle_length=8,
        leak_node_name=env.leak_node_name, vmin=vmin, vmax=vmax, step=5
    )
    plt.show()

    #for n_epi in range(episodes):
        #s = env.reset()
        #done, score = False, 0.0

        #print("\n=== 🌊 Simulazione SENZA LEAK ===")
        #s_no_leak = env.reset(with_leak=False)
        #plot_topological_cycles(env.wn, env.data, env.f, max_cycle_length=8, leak_node_name=None)

        #print("\n=== 💧 Simulazione CON LEAK ===")
        #s_with_leak = env.reset(with_leak=True)
        #plot_topological_cycles(env.wn, env.data, env.f, max_cycle_length=8, leak_node_name=env.leak_node_name)
        



    """
        while not done:
            a = q.sample_action(s, env.global_step)
            s_prime, r, done, _ = env.step(a)
            memory.put((s, a, r, s_prime, 0.0 if done else 1.0))
            s = s_prime
            score += r

        if n_epi % target_update_interval == 0 and n_epi != 0:
            avg_score = sum(score_list[-target_update_interval:]) / target_update_interval
            print(f"Ep {n_epi}, avg score {avg_score:.2f}")
            q_target.load_state_dict(q.state_dict())

        if memory.size() > train_interval:
            train(q, q_target, memory, optimizer)
            memory.clear()

        score_list.append(score)

    plt.plot(score_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("WNTR DQN Training (Dynamic Step Sim)")
    plt.grid()
    plt.show()
    """

def plot_topological_cycles(wn, data, f, max_cycle_length=8, leak_node_name=None, vmin=None, vmax=None, step=None):
    """
    Mostra il grafo WNTR con i cicli topologici colorati in base al flusso netto.
    - Evidenzia nodo leak e tubi chiusi.
    """
    import matplotlib.pyplot as plt
    from wntr.network.elements import LinkStatus
    import networkx as nx
    import numpy as np
    from main_dyn_topologyknown_01 import func_gen_B2_lu, plot_cell_complex

    # 1️⃣ Crea grafo NetworkX (come prima)
    G = nx.Graph()
    edges = data.edge_index.cpu().T.numpy()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edges)
    

    # 2️⃣ Calcola B1, B2 e cicli
    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length)

    # 3️⃣ Calcola flusso netto per poligono
    f_polygons = compute_polygon_flux_from_f_B2(f, B2)
    flux_abs = np.abs(f_polygons.flatten())

    if vmin is None:
        vmin = flux_abs.min()
    if vmax is None:
        vmax = flux_abs.max()

    from matplotlib import colors
    flux_norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # 4️⃣ Ottieni coordinate dei nodi
    coords = np.array([wn.get_node(name).coordinates for name in wn.node_name_list])

    # 5️⃣ Plot topologico (colorato per flusso)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("plasma")

    for ci, cycle in enumerate(selected_cycles):
        pts = coords[cycle]
        poly = np.vstack([pts, pts[0]])
        color = cmap(flux_norm(flux_abs[ci])) if ci < len(flux_abs) else "gray"
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor="none", alpha=0.6)

        centroid_x = np.mean(poly[:, 0])
        centroid_y = np.mean(poly[:, 1])

        # Etichetta del flusso (in m³/s)
        flux_val = f_polygons[ci, 0]
        ax.text(centroid_x, centroid_y, f"{flux_val:.3f}",
                color="black", fontsize=8, ha="center", va="center",
                fontweight="bold", zorder=12)
        

    # 6️⃣ Disegna gli archi e i nodi base
    for u, v in G.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.plot([x0, x1], [y0, y1], color="k", linewidth=1)

    ax.scatter(coords[:, 0], coords[:, 1], s=30, facecolor="w", edgecolor="k", zorder=3)

    # 7️⃣ Tubi chiusi → X rosse
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        if pipe.status != LinkStatus.Open:
            u, v = pipe.start_node, pipe.end_node
            x_mid = (u.coordinates[0] + v.coordinates[0]) / 2
            y_mid = (u.coordinates[1] + v.coordinates[1]) / 2
            ax.scatter(x_mid, y_mid, color="red", marker="x", s=120, zorder=5)

    # 8️⃣ Nodo leak
    if leak_node_name is not None and leak_node_name in wn.node_name_list:
        leak_node = wn.get_node(leak_node_name)
        x, y = leak_node.coordinates
        ax.scatter(x, y, color="red", s=25, marker="o",
                    edgecolor="black", linewidths=1.0, zorder=10)
        ax.text(x + 2, y + 2, f"LEAK",
                color="red", fontsize=10, fontweight="bold", zorder=11)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, label="Flusso netto per poligono [m³/s]")
    ax.set_title(f"Flowrate poligoni - step {step}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return vmin,vmax
    


def compute_polygon_flux_from_f_B2(f, B2):
    """
    Calcola il flusso netto per ciascun poligono
    in base al vettore dei flussi (f) e alla matrice topologica B2.

    f:  [Nedge x 1]  vettore dei flussi (in m³/s)
    B2: [Nedge x Npolygons] matrice topologica (da func_gen_B2_lu)

    Ritorna:
        f_polygons: [Npolygons x 1] vettore dei flussi per poligono
    """


    # Moltiplicazione topologica: B2' * f
    f_polygons = B2.T @ f

    return f_polygons



if __name__ == "__main__":
    run_wntr_experiment(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Grid.inp")

