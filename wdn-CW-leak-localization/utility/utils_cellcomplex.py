
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import qr

# ============================================================
#  CODICE DI LUCIA
# ============================================================

NUM_NODES = 80  # Numero di nodi
NUM_EDGES = 100  # Numero di archi

def plot_cell_complex(G, coords, selected_cycles,
                      figsize=(6,6),
                      cmap='turbo',
                      edge_color='k',
                      node_size=30,
                      node_facecolor='w',
                      node_edgecolor='k',
                      annotate=True,
                      annot_fontsize=8):
    """
    Plot the graph G with its selected cycle‐cells overlaid.

    Args:
      G               networkx.Graph
      coords          np.ndarray of shape (n_nodes,2): (x,y) positions
      selected_cycles list of cycles, each a list of node‐indices
      figsize         tuple: matplotlib figure size
      cmap            string or Colormap: for the cycle fill
      edge_color      str: color for graph edges
      node_size       int: size for node markers
      node_facecolor  str: fill color for nodes
      node_edgecolor  str: outline color for nodes
      annotate        bool: whether to label nodes by index
      annot_fontsize  int: font size for labels
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_cycles = len(selected_cycles)

    # 1) cycle fills
    # a random color value per cycle, normalized 0..1
    rand_vals = np.random.rand(n_cycles)
    norm = plt.Normalize(vmin=rand_vals.min(), vmax=rand_vals.max())
    cmap_obj = plt.get_cmap(cmap)

    for ci, cycle in enumerate(selected_cycles):
        pts = coords[cycle]  # list of (x,y) for this cycle
        # make sure polygon is closed
        poly = np.vstack([pts, pts[0]])
        ax.fill(poly[:,0], poly[:,1],
                facecolor=cmap_obj(norm(rand_vals[ci])),
                edgecolor='none',
                alpha=0.6)

    # 2) graph edges
    for u, v in G.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1)

    # 3) nodes
    ax.scatter(coords[:,0], coords[:,1],
               s=node_size,
               facecolor=node_facecolor,
               edgecolor=node_edgecolor,
               zorder=3)

    # 4) optional annotations
    if annotate:
        for idx, (x, y) in enumerate(coords):
            ax.text(x, y, str(idx),
                    fontsize=annot_fontsize,
                    ha='center', va='center', zorder=4)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig, ax

def find_incremental_cycle_basis(graph):
    """
    Calcola la base dei cicli in modo incrementale:
    - Prima cerca i cicli di lunghezza 3 (triangoli),
      poi quelli di lunghezza 4, e così via.
    - Aggiunge un ciclo alla base solo se risulta linearmente indipendente
      (in GF(2)) rispetto ai cicli già trovati.
    - L'enumerazione dei cicli viene interrotta non appena la base è completa.    Restituisce una lista di cicli (ognuno come lista di nodi, dove il ciclo
    è implicito: l'ultimo nodo si collega al primo).
    """
    # Costruisce la mappa degli archi (rappresentati come frozenset) a indici.
    edge_index = {frozenset({u, v}): i for i, (u, v) in enumerate(graph.edges())}
    num_edges = len(edge_index)    # La base dei cicli (lista di cicli come liste di nodi)
    basis = []
    # La base in forma di bitmask (vettori in GF(2))
    basis_bitmasks = []    
    
    def cycle_bitmask(cycle):
        """
        Dato un ciclo (lista di nodi) calcola il corrispondente bitmask.
        Si assume che l'arco tra l'ultimo nodo e il primo sia presente.
        """
        bitmask = 0
        n: int = len(cycle)
        for i in range(n):
            # L'arco è rappresentato in maniera non orientata
            edge = frozenset({cycle[i], cycle[(i+1) % n]})
            idx = edge_index[edge]
            bitmask ^= (1 << idx)
        return bitmask
      
    def add_vector_to_basis(vec, basis_bitmasks):
        """
        Riduce il vettore candidato 'vec' usando la base corrente (lista di bitmask).
        Se il vettore ridotto non è nullo, lo aggiunge alla base e restituisce True.
        """
        for b in basis_bitmasks:
            pivot = b.bit_length() - 1  # indice del bit più significativo
            if vec & (1 << pivot):
                vec ^= b
        if vec != 0:
            basis_bitmasks.append(vec)
            # Manteniamo la base ordinata per pivot decrescente.
            basis_bitmasks.sort(key=lambda x: x.bit_length(), reverse=True)
            return True
        return False    # Dimensione teorica della base: |E| - |V| + (numero di componenti connesse)
    
    
    required_basis_size = num_edges - graph.number_of_nodes() + nx.number_connected_components(graph)    # Ordiniamo i nodi per garantire determinismo nell'enumerazione
    nodes = sorted(graph.nodes())    # Per evitare duplicati durante l'enumerazione dei cicli
    found_cycles = set()  # memorizza tuple (canonical) dei nodi del ciclo    
    
    def dfs(start, current, depth, L, path, visited):
        """
        Cerca cicli semplici di lunghezza esattamente L a partire da 'start'.
        Si garantisce che i nodi visitati siano >= start per canonicalità.
        """
        if depth == L:
            # Se il nodo corrente è adiacente a start, abbiamo trovato un ciclo
            if start in graph[current]:
                cycle = path[:]  # il ciclo è dato dai nodi in path (l'arco finale chiude il ciclo)
                tup = tuple(cycle)
                if tup not in found_cycles:
                    found_cycles.add(tup)
                    yield cycle
            return        
        for neighbor in graph[current]:
            # Per evitare duplicati, consideriamo solo neighbor >= start
            if neighbor < start:
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            yield from dfs(start, neighbor, depth + 1, L, path, visited)
            path.pop()
            visited.remove(neighbor)    # Enumeriamo cicli per lunghezza L = 3, 4, ... fino a |V|


    for L in range(3, graph.number_of_nodes() + 1):
        for start in nodes:
            visited = {start}
            path = [start]
            for cycle in dfs(start, start, 1, L, path, visited):
                # Calcola il bitmask del ciclo trovato
                bitmask = cycle_bitmask(cycle)
                # Se il ciclo è linearmente indipendente rispetto alla base attuale, lo aggiunge.
                if add_vector_to_basis(bitmask, basis_bitmasks):
                    basis.append(cycle)
                    if len(basis_bitmasks) == required_basis_size:
                        return basis
    return basis


def get_all_cycles_undirected(G, max_length=None):
    cycles = find_incremental_cycle_basis(G)
    if max_length is not None:
        cycles = [c for c in cycles if len(c) <= max_length]
    return cycles


def func_gen_B2_lu(G, max_cycle_length):
    """
    Core logic only (no plotting):
      - find all simple cycles up to length `max_cycle_length`
      - build B1 (node*edge incidence)
      - build raw B2 (edge*cycle incidence)
      - pick an independent subset of cycles via QR-pivoting
    Returns:
      B1               np.ndarray shape (n_nodes, n_edges)
      B2_final         np.ndarray shape (n_edges, n_selected_cycles)
      selected_cycles  list of selected cycle lists (node‐index lists)
    """
    cycles = get_all_cycles_undirected(G, max_length=max_cycle_length)

    #print(f"Found {len(cycles)} cycles up to length {max_cycle_length} in the graph.")

    edge_list = [tuple(sorted(e)) for e in G.edges()]
    n_edges  = len(edge_list)
    n_cycles = len(cycles)
    B2 = np.zeros((n_edges, n_cycles), dtype=int)

    for c_idx, cycle in enumerate(cycles):
        extended = cycle + [cycle[0]]
        for j in range(len(cycle)):
            u, v = extended[j], extended[j+1]
            sorted_e = tuple(sorted((u, v)))
            row = edge_list.index(sorted_e)
            # +1 if (u→v) matches sorted order, else −1
            B2[row, c_idx] = 1 if (u, v) == sorted_e else -1

    # — Pick independent columns via QR pivoting —
    # 1) sort columns by nonzeros (short cycles first)
    nnz = np.count_nonzero(B2, axis=0)
    idx_sorted = np.argsort(nnz)
    mat_sorted = B2[:, idx_sorted]

    # 2) QR with pivoting
    Q, R, piv = qr(mat_sorted, pivoting=True, mode='economic')

    # 3) determine numerical rank
    tol = np.abs(R).max() * max(R.shape) * np.finfo(R.dtype).eps
    rank = np.sum(np.abs(np.diag(R)) > tol)

    # 4) pick the first `rank` pivots
    pivotcols = piv[:rank]
    original_indices = [idx_sorted[i] for i in pivotcols]

    B2_final      = B2[:, original_indices]
    selected_cycles = [cycles[i] for i in original_indices]

    # — Build B1: node*edge incidence (arbitrary orientation) —
    n_nodes = G.number_of_nodes()
    B1 = np.zeros((n_nodes, n_edges), dtype=int)

    for e_idx, (u, v) in enumerate(G.edges()):
        B1[u, e_idx] =  1
        B1[v, e_idx] = -1

    return B1, B2_final, selected_cycles

