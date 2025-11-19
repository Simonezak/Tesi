import collections
import random
from torch_geometric.data import Data


buffer_limit = 5000

class ReplayBuffer:
    def __init__(self, maxlen=buffer_limit):
        self.buffer = collections.deque(maxlen=maxlen)

    def put(self, transition):
        """
        transition = (state: Data, action: int, reward: float, next_state: Data, done_mask: float)
        """
        self.buffer.append(transition)

    def sample(self, n):
        """
        Ritorna n transizioni random come liste
        - batch_s: lista di Data (stati)
        - batch_a: lista di int (azioni)
        - batch_r: lista di float (reward)
        - batch_s_prime: lista di Data (next states)
        - batch_done: lista di float (done mask)
        """
        mini_batch = random.sample(self.buffer, n)

        batch_s, batch_a, batch_r, batch_s_prime, batch_done = [], [], [], [], []
        for s, a, r, s_prime, done in mini_batch:
            assert isinstance(s, Data) and isinstance(s_prime, Data), "Gli stati devono essere grafi PyG (Data)"
            batch_s.append(s)
            batch_a.append(a)
            batch_r.append(r)
            batch_s_prime.append(s_prime)
            batch_done.append(done)

        return batch_s, batch_a, batch_r, batch_s_prime, batch_done

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
