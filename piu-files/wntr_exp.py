import torch
import torch.optim as optim
import random
import networkx as nx
from torch_geometric.utils import to_networkx

from DQN.gnn import DQNGNN
from DQN.replay_buffer import ReplayBuffer
from DQN.train_utils import seed_torch, train, device
from data_utils.wntr_to_pyg import build_pyg_from_wntr
from actions import open_pipe, close_pipe, close_all_pipes, noop
import matplotlib.pyplot as plt

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, max_steps=5, hydraulic_timestep=300):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.sim = None
        self.wn = None
        self.results = None
        self.current_step = 0
        self.done = False

    def reset(self):
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)

        # Crea simulatore interattivo e inizializza
        self.sim = InteractiveWNTRSimulator(self.wn)
        self.sim.init_simulation(global_timestep=self.hydraulic_timestep,
                                 duration=self.max_steps * self.hydraulic_timestep)

        self.sim.step_sim()  # Crea lo stato iniziale
        self.results = self.sim.get_results()
        print(self.results.node["pressure"].iloc[-1])

        self.data, *_ = build_pyg_from_wntr(self.wn, self.results, -1)
        self.num_pipes = int(self.data.num_pipes)
        self.action_dim = 2 * self.num_pipes  # 0=close, 1=open per ciascun tubo
        self.current_step = 0
        print(self.current_step)
        self.done = False
        return self.data
    
    def step(self, action_index):

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

        elif action_index == 2 * self.num_pipes + 1:
            close_all_pipes(self.sim)

        else:
            raise ValueError(f"Azione fuori range: {action_index}")

        # Avanza la simulazione di un passo
        self.sim.step_sim()
        self.results = self.sim.get_results()
        print(self.results.node["pressure"].iloc[-1])

        next_state, *_ = build_pyg_from_wntr(self.wn, self.results, -1)

        # Reward: pressione media vicina a 50
        pressures = next_state.x[:, 2].mean().item()
        reward = -abs(pressures - 50.0)

        self.current_step += 1
        print(self.current_step)
        done = self.current_step >= self.max_steps or self.sim.is_terminated()
        return next_state, reward, done, {}
    

def plot_network_state(self, title=""):
    """
    Mostra la rete idrica corrente:
    - Nodi colorati per pressione
    - X rossa su archi (tubi) chiusi
    """
    data = self.data
    if data is None:
        print("⚠️ Nessun grafo disponibile per la visualizzazione.")
        return

    # Converti PyG → NetworkX
    G = to_networkx(data, to_undirected=True)

    # Posizioni nodi (se disponibili da WNTR, meglio; altrimenti usa layout generico)
    pos = {}
    if hasattr(self.wn, "node_name_list"):
        for i, name in enumerate(data.pipe_names):
            node = self.wn.get_node(name)
            pos[i] = node.coordinates
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    # Colori nodi per pressione
    pressures = data.x[:, 2].cpu().numpy()
    node_color = pressures

    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=60,
        node_color=node_color,
        cmap="coolwarm",
        edge_color="lightgray",
    )

    # X rosse sui tubi chiusi
    if hasattr(data, "pipe_open_mask"):
        for i, is_open in enumerate(data.pipe_open_mask.cpu().numpy()):
            if is_open < 0.5:  # tubo chiuso
                edge_idx = data.pipe_edge_idx[i].item()
                u, v = data.edge_index[:, edge_idx].tolist()
                x_mid = (pos[u][0] + pos[v][0]) / 2
                y_mid = (pos[u][1] + pos[v][1]) / 2
                plt.scatter(x_mid, y_mid, color="red", marker="x", s=120, zorder=5)

    plt.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), label="Pressione")
    plt.title(title or "Stato rete idrica (rosso = tubo chiuso)")
    plt.axis("equal")
    plt.show()


def run_wntr_experiment(inp_path):
    seed = 42
    random.seed(seed)
    seed_torch(seed)

    env = WNTREnv(inp_path, max_steps=5)

    episodes = 3
    learning_rate = 5e-4
    target_update_interval = 5
    train_interval = 1000

    q = DQNGNN().to(device)
    q_target = DQNGNN().to(device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    score_list = []

    for n_epi in range(episodes):
        s = env.reset()
        done, score = False, 0.0
        epsilon = max(0.01, 0.1 - 0.01 * (n_epi / 50))

        _ = q(s, debug=False)  # solo per validare le dimensioni

        while not done:
            a = q.sample_action(s, epsilon)
            s_prime, r, done, _ = env.step(a)
            memory.put((s, a, r, s_prime, 0.0 if done else 1.0))
            s = s_prime
            score += r

        if n_epi % target_update_interval == 0 and n_epi != 0:
            avg_score = sum(score_list[-target_update_interval:]) / target_update_interval
            print(f"Ep {n_epi}, avg score {avg_score:.2f}, eps {epsilon:.2f}")
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



if __name__ == "__main__":
    run_wntr_experiment(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\Dynamic-WNTR-dynwntr\examples\networks\Net3.inp")

