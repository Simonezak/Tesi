from DQN.gnn import DQNGNN
from DQN.replay_buffer import ReplayBuffer
from DQN.train_utils import seed_torch, train, device
from data_utils.wntr_to_pyg import run_wntr_simulation, build_pyg_from_wntr
from wntr.network.elements import LinkStatus

import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from actions import open_pipe, close_pipe, close_all_pipes, noop
import wntr


class WNTREnv:
    def __init__(self, inp_path, max_steps=20):
        self.inp_path = inp_path
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.wn, self.results, self.t_idx = run_wntr_simulation(self.inp_path)
        self.current_step = 0
        self.done = False

        # apri tutti i tubi
        for pipe_name in self.wn.pipe_name_list:
            pipe = self.wn.get_link(pipe_name)
            pipe.initial_status = LinkStatus.Open

        self.data, *_ = build_pyg_from_wntr(self.wn, self.results, self.t_idx)
        self.num_pipes = int(self.data.num_pipes)
        self.action_dim = 2 * self.num_pipes
        return self.data

    def step(self, action_index):
        self.current_step += 1

        if action_index < 2 * self.num_pipes:
            pipe_id = action_index // 2
            act = action_index % 2  # 0=close, 1=open
            pipe_name = self.data.pipe_names[pipe_id]

            if act == 0:
                close_pipe(self.wn, pipe_name)
            else:
                open_pipe(self.wn, pipe_name)

        elif action_index == 2 * self.num_pipes:
            noop(self.wn)

        elif action_index == 2 * self.num_pipes + 1:
            close_all_pipes(self.wn)

        else:
            raise ValueError(f"Azione fuori range: {action_index}")

        # nuova simulazione
        sim = wntr.sim.WNTRSimulator(self.wn)
        self.results = sim.run_sim()
        self.t_idx = -1
        next_state, *_ = build_pyg_from_wntr(self.wn, self.results, self.t_idx)

        pressures = next_state.x[:, 2].mean().item()
        reward = -abs(pressures - 50.0)

        done = self.current_step >= self.max_steps
        return next_state, reward, done, {}
    

    def plot_network_state(self, title=""):
        pressures = self.results.node["pressure"].iloc[self.t_idx]
        plt.figure(figsize=(10, 8))
        wntr.graphics.network.plot_network(
            self.wn,
            node_attribute=pressures,
            node_size=30,
            node_range=[pressures.min(), pressures.max()],
            add_colorbar=True,
        )
        plt.title(title)
        plt.show()

    def plot_multiple_steps(self, results_list, step_list, actions, titles=None):
        """
        Mostra più stati della rete in un'unica figura e colora di rosso
        il tubo su cui è stata fatta l'azione.
        """
        n = len(step_list)
        plt.figure(figsize=(6*n, 6))

        for i, step in enumerate(step_list):
            pressures = results_list[i].node["pressure"].iloc[step]
            ax = plt.subplot(1, n, i+1)

            # base: tutti i tubi grigi

            # prendi l'action index
            a = actions[i]
            if isinstance(a, int) and a < 2*self.num_pipes:
                pipe_id = a // 2
                pipe_name = self.data.pipe_names[pipe_id]

                # opzionale: aggiungi freccia/annotazione
                link = self.wn.get_link(pipe_name)
                u, v = link.start_node, link.end_node
                x_mid = (u.coordinates[0] + v.coordinates[0]) / 2
                y_mid = (u.coordinates[1] + v.coordinates[1]) / 2
                ax.annotate("azione", xy=(x_mid, y_mid), xytext=(x_mid+20, y_mid+20),
                            arrowprops=dict(facecolor="red", shrink=0.05))

            wntr.graphics.network.plot_network(
                self.wn,
                node_attribute=pressures,
                node_size=30,
                node_range=[pressures.min(), pressures.max()],
                add_colorbar=False,
                ax=ax,
            )

            if titles is not None:
                ax.set_title(titles[i])

        plt.tight_layout()
        plt.show()





def run_wntr_experiment(inp_path):
    seed = 42
    random.seed(seed)
    seed_torch(seed)

    env = WNTREnv(inp_path)

    # RL params
    episodes = 20
    learning_rate = 5e-4
    target_update_interval = 5
    train_interval = 1000

    # la rete produce dinamicamente (1, 2*P) dal grafo
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

        # stampa forme richieste:
        _ = q(s, debug=False)

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
        plt.title("WNTR DQN Training (azioni per pipe)")
        plt.grid()
        plt.show()


def run_wntr_experiment_exp(inp_path):
    seed = 42
    random.seed(seed)
    seed_torch(seed)

    env = WNTREnv(inp_path)
    q = DQNGNN().to(device)

    s = env.reset()
    epsilon = 0.1

    print("🔎 Avvio esperimento di debug con primi 3 step...")

    results_list = []
    step_list = []
    titles = []
    actions = []

    for i in range(3):
        a = q.sample_action(s, epsilon)
        s_prime, r, done, _ = env.step(a)

        results_list.append(env.results)
        step_list.append(env.t_idx)
        titles.append(f"Step {i+1} - Action {a}")
        actions.append(a)

        s = s_prime

    # 🔹 ora passi anche actions
    env.plot_multiple_steps(results_list, step_list, actions, titles)





if __name__ == "__main__":
    run_wntr_experiment_exp(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\WNTR-main\WNTR-main\examples\networks\Net3.inp")
