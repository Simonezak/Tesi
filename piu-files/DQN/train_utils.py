import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

gamma = 0.98
batch_size = 32
num_update_steps = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train(q, q_target, memory, optimizer):
    """
    Aggiorna la rete Q rispetto alla rete target usando il replay buffer.
    """

    for _ in range(num_update_steps):
        batch_s, batch_a, batch_r, batch_s_prime, batch_done = memory.sample(batch_size)

        # Batch dei grafi
        batch_s = Batch.from_data_list(batch_s).to(device)
        batch_s_prime = Batch.from_data_list(batch_s_prime).to(device)

        # Tensor version di azioni, reward e done
        batch_a = torch.tensor(batch_a, dtype=torch.long, device=device).unsqueeze(1)   # (B,1)
        batch_r = torch.tensor(batch_r, dtype=torch.float, device=device).unsqueeze(1) # (B,1)
        batch_done = torch.tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1) # (B,1)

        # Forward pass rete Q
        q_out = q(batch_s)            # (B, action_dim)
        q_a = q_out.gather(1, batch_a) # seleziona Q(s,a)

        # Forward rete target
        q_prime = q_target(batch_s_prime)        # (B, action_dim)
        max_q_prime = q_prime.max(1)[0].unsqueeze(1)  # max_a' Q(s', a')

        # Bellman target
        target = batch_r + gamma * max_q_prime * batch_done

        # Loss Huber
        loss = F.smooth_l1_loss(q_a, target)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        optimizer.step()
