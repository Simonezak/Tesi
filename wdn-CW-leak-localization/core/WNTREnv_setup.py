
import torch
import numpy as np

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator


class WNTREnv:
    def __init__(self, inp_path, num_steps=5, hydraulic_timestep=3600):
        self.inp_path = inp_path
        self.num_steps = num_steps
        self.hydraulic_timestep = hydraulic_timestep
        self.sim = None
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.results = None

        

    def reset(self, num_leaks=2):


        # Crea rete
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)
        self.sim = InteractiveWNTRSimulator(self.wn)


        # Aggiungi un leak
        self.leak_node_names = []
        self.leak_start_step = None
            
        if num_leaks > 0:
            # Prendiamo solo le junctions
            junctions = [
                name for name, node in self.wn.nodes()
                if isinstance(node, wntr.network.elements.Junction)
            ]

            num = min(num_leaks, len(junctions))
            self.leak_node_names = np.random.choice(
                junctions, size=num, replace=False
            ).tolist()

            # Step di inizio leak
            self.leak_start_step = np.random.randint(10, 26)

            print(f"[LEAK] Nodi selezionati per la perdita: {self.leak_node_names}")
            print(f"[LEAK] Il leak inizierà allo step {self.leak_start_step}")
        else:
            print("[INIT] Episodio senza leak")
            
        self.sim.init_simulation(
            global_timestep=self.hydraulic_timestep,
            duration=self.num_steps * self.hydraulic_timestep
        )

        return


def build_adj_matrix(wn):
    """
    Costruisce:
    - adjacency matrix [1, N, N]
    - node2idx, idx2node
    usando SOLO le junctions.
    """

    # --- seleziona solo junctions ---
    junction_names = [
        name for name, node in wn.nodes()
        if isinstance(node, wntr.network.elements.Junction)
    ]

    node2idx = {name: i for i, name in enumerate(junction_names)}
    idx2node = {i: name for name, i in node2idx.items()}
    N = len(junction_names)

    # --- adjacency matrix ---
    adj = torch.zeros((N, N), dtype=torch.float32)

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        u_name = pipe.start_node_name
        v_name = pipe.end_node_name

        # scarta pipe che toccano tank / reservoir
        if u_name not in node2idx or v_name not in node2idx:
            continue

        u = node2idx[u_name]
        v = node2idx[v_name]

        adj[u, v] = 1.0
        adj[v, u] = 1.0   # grafo non orientato

    # aggiungi dimensione batch
    adj_matrix = adj.unsqueeze(0)

    return adj_matrix, node2idx, idx2node

