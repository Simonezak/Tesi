import torch
import torch.optim as optim
import random
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx

from DQN.gnn import DQNGNN
from DQN.replay_buffer import ReplayBuffer
from DQN.train_utils import seed_torch, train, device
from data_utils.wntr_to_pyg import build_pyg_from_wntr, build_nx_graph_from_wntr
from actions import open_pipe, close_pipe, close_all_pipes, noop
import matplotlib.pyplot as plt
from main_dyn_topologyknown_01 import func_gen_B2_lu
from topological import compute_polygon_flux, get_inital_polygons_flux_limits, plot_cell_complex_flux, construct_matrix_f

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
            """
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]
            self.leak_node_name = np.random.choice(junctions)
            """
            self.leak_node_name = "J1"
            print(f"[LEAK] Nodo selezionato per la perdita: {self.leak_node_name}")

            # parametri leak
            area = 5e-1 * (0.3 ** 2) * np.random.uniform(0.9, 1.1)
            self.sim.start_leak(self.leak_node_name, leak_area=area, leak_discharge_coefficient=0.75)
            
            print(f"[LEAK] Aggiunta perdita con area={area:.3e} m²")
        else:
            print("[INIT] Nessuna perdita inserita in questo episodio.")

        # ===============================
        # 3️⃣ Inizializza simulatore WNTR
        # ===============================
        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.max_steps * self.hydraulic_timestep
        )

        return

    
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

    # ---------------------------
    # 1️⃣ Setup ambiente
    # ---------------------------
    print("\n=== 💧 Simulazione WNTR (Dynamic Topology + Leak) ===")
    env = WNTREnv(inp_path, max_steps=5, hydraulic_timestep=3600)
    env.reset(with_leak=True)
    wn = env.wn
    sim = env.sim

    # ---------------------------
    # 2️⃣ Esegui alcuni step di simulazione
    # ---------------------------
    print("\n Esecuzione simulazione")
    for step in range(1, env.max_steps + 1):
        print(f" Step {step}/{env.max_steps}")
        sim.step_sim()
        results = sim.get_results()

    env.results = results  # aggiorna l’ambiente

    # ---------------------------
    # 3️⃣ Costruisci il grafo NetworkX e calcola B1, B2, cicli
    # ---------------------------
    G, coords = build_nx_graph_from_wntr(wn, results)

    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)


    # Calcolo matrice dei flussi
    f = construct_matrix_f(wn, results)

    # Calcolo flusso per poligono e limiti iniziali per confrontare
    f_polygons = compute_polygon_flux(f, B2)
    vmin, vmax = get_inital_polygons_flux_limits(f_polygons)

    # Ottengo Coordinate e leak node per la funzione
    leak_node = wn.get_node(env.leak_node_name)
    
    # Visualizza con la nuova funzione
    plot_cell_complex_flux(
        G,
        coords,
        selected_cycles,
        f_polygons=f_polygons,
        vmin=vmin,
        vmax=vmax,
        leak_node=leak_node,
        step=step,
        figsize=(8, 8),
        cmap="plasma",
        node_size=40,
        annotate=True,
    )
    plt.show()

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



if __name__ == "__main__":
    run_wntr_experiment(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Networks-found\Grid.inp")

