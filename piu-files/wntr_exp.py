

from DQN.gnn import DQNGNN
from DQN.replay_buffer import ReplayBuffer
from DQN.train_utils import seed_torch, train, device
from data_utils.wntr_to_pyg import run_wntr_simulation, build_pyg_from_wntr
from wntr.network.elements import LinkStatus

import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt



# -------------------------
# Ambiente custom WNTR
# -------------------------
class WNTREnv:
    def __init__(self, inp_path, target_pipe, max_steps=20):
        self.inp_path = inp_path
        self.target_pipe = target_pipe
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        import wntr
        self.wn, self.results, self.t_idx = run_wntr_simulation(self.inp_path)
        self.current_step = 0
        self.done = False

        # ⚠️ inizializza tutti i tubi come aperti
        for pipe_name in self.wn.pipe_name_list:
            pipe = self.wn.get_link(pipe_name)
            pipe.initial_status = LinkStatus.Open

        self.data, *_ = build_pyg_from_wntr(self.wn, self.results, self.t_idx)
        return self.data

    def step(self, action):
        import wntr
        self.current_step += 1

        # Modifica stato tubo target
        pipe = self.wn.get_link(self.target_pipe)
        if action == 0:
            pipe.initial_status = LinkStatus.Closed
        elif action == 1:
            pipe.initial_status = LinkStatus.Open

        # Nuova simulazione
        sim = wntr.sim.WNTRSimulator(self.wn)
        self.results = sim.run_sim()
        self.t_idx = -1
        next_state, *_ = build_pyg_from_wntr(self.wn, self.results, self.t_idx)

        # Reward: pressione media vicina a 50
        pressures = next_state.x[:, 2].mean().item()
        reward = -abs(pressures - 50)

        done = self.current_step >= self.max_steps
        return next_state, reward, done, {}
    

def run_wntr_experiment(inp_path):
# Impostazioni random seed
    seed = 42
    random.seed(seed)
    seed_torch(seed)

    # Scegli un tubo target (per test prendiamo il primo)
    import wntr
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = 3600    # 1 ora in secondi
    wn.options.time.hydraulic_timestep = 300  # 5 minuti in secondi

    target_pipe = wn.pipe_name_list[0]  # 👈 puoi cambiare quale tubo controllare
    print(f"Userò come tubo controllabile: {target_pipe}")

    # Ambiente con 2 azioni (apri/chiudi tubo)
    env = WNTREnv(inp_path, target_pipe=target_pipe)

    # Parametri RL
    action_dim = 2
    episodes = 20
    learning_rate = 0.0005
    target_update_interval = 5
    train_interval = 1000

    # Inizializza reti DQN
    q = DQNGNN(action_dim=action_dim).to(device)
    q_target = DQNGNN(action_dim=action_dim).to(device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    score_list = []

    for n_epi in range(episodes):
        s = env.reset()
        done, score = False, 0.0
        epsilon = max(0.01, 0.1 - 0.01 * (n_epi / 50))  # decrescente

        while not done:
            a = q.sample_action(s, epsilon)
            s_prime, r, done, _ = env.step(a)

            memory.put((s, a, r, s_prime, 0.0 if done else 1.0))
            s = s_prime
            score += r

        # Aggiorna target network ogni X episodi
        if n_epi % target_update_interval == 0 and n_epi != 0:
            avg_score = sum(score_list[-target_update_interval:]) / target_update_interval
            print(f"Ep {n_epi}, avg score {avg_score:.2f}, eps {epsilon:.2f}")
            q_target.load_state_dict(q.state_dict())

        # Allena rete se ci sono abbastanza esperienze
        if memory.size() > train_interval:
            train(q, q_target, memory, optimizer)
            memory.clear()

        score_list.append(score)

    plt.plot(score_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("WNTR DQN Training")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_wntr_experiment(inp_path=r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\WNTR-main\WNTR-main\examples\networks\Net3.inp")