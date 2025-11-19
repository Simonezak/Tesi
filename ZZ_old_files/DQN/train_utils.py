import torch
import torch.nn.functional as F

gamma = 0.98
batch_size = 32          # quante transizioni per update (elaborate una per volta)
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
    NOTA: elabora le transizioni UNA ALLA VOLTA (niente Batch.from_data_list),
          per supportare azioni (2*P) dinamiche dipendenti dal grafo.
    """
    q.train()
    q_target.eval()

    for _ in range(num_update_steps):
        if memory.size() < batch_size:
            break

        # prendo un "batch" di indici, ma le elaboro singolarmente
        batch_s, batch_a, batch_r, batch_s_prime, batch_done = memory.sample(batch_size)

        losses = []
        for s, a, r, s_prime, done in zip(batch_s, batch_a, batch_r, batch_s_prime, batch_done):
            # s, s_prime: Data (grafi PyG)
            # a: int (indice nell'intervallo [0, 2*P-1])
            # r: float
            # done: float (1.0 se non terminale, 0.0 se terminale nel tuo codice)

            # Q(s, ·) e selezione Q(s,a)
            q_out = q(s)                  # (1, 2*P)
            q_a = q_out[0, a]             # scalar

            # Q_target(s', ·) e max_a' Q(s',a')
            with torch.no_grad():
                q_prime = q_target(s_prime)          # (1, 2*P')
                max_q_prime = q_prime.max(dim=1)[0]  # (1,)
                y = torch.tensor([r], dtype=torch.float, device=q_out.device) + \
                    gamma * max_q_prime * torch.tensor([done], dtype=torch.float, device=q_out.device)

            loss = F.smooth_l1_loss(q_a.unsqueeze(0), y)
            losses.append(loss)

        if not losses:
            continue

        loss_mean = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        optimizer.step()
