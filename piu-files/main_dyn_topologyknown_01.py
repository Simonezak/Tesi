import decimal
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.linalg import qr
import cvxpy as cp

from lucia_code import find_incremental_cycle_basis


NUM_NODES = 80  # Numero di nodi
NUM_EDGES = 100  # Numero di archi
NUM_GRAPHS = 50  # Numero di grafi da generare
LAMBDA_MAX = 1.0  # Valore massimo di lambda per la normalizzazione
NUM_HOLES = 15
TIME_INSTANTS = 20

NUM_ANOMALIES = 8
NUM_SPARSE = 35
NUM_REPETITIONS = 10

PLOT_GRAPH = True
SAVE_FIG = False

def frange(x, y, jump):
    jump = decimal.Decimal(jump)
    x = decimal.Decimal(x)
    ret = []
    while x <= y:
        ret.append(float(x))
        x += jump
    return ret
    
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

def generate_random_graph(num_nodes, num_edges):
    # Generazione dei nodi casuali
    coords_matrix = np.random.rand(num_nodes, 2)

    # Triangolazione di Delaunay
    tri = Delaunay(coords_matrix)
    triangles = tri.simplices
    all_edge_sets = set()
    while len(all_edge_sets) < num_edges:
        # Selezione casuale di triangoli
        selected_triangles = np.random.choice(len(triangles), size=min(num_edges, len(triangles)), replace=False)
        for idx in selected_triangles:
            tri = triangles[idx]
            for u, v in itertools.combinations(tri, 2):
                all_edge_sets.add(tuple(sorted((u, v))))
    #for tri in triangles:
    #    for u, v in itertools.combinations(tri, 2):
    #        all_edge_sets.add(tuple(sorted((u, v))))
    full_edges = np.array(list(all_edge_sets))  # shape (M,2)

    # Selezione casuale degli archi
    rand_idx = np.random.choice(full_edges.shape[0], num_edges, replace=False)
    edges = full_edges[rand_idx]

    # Creazione della matrice di adiacenza
    graph = nx.Graph()
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    adj_matrix = nx.to_numpy_array(graph)

    return graph, coords_matrix, adj_matrix

def plot_graph(graph, coords_matrix):
    plt.figure()
    nx.draw(graph, pos=coords_matrix, with_labels=False, node_size=50, edge_color='b', alpha=0.5)
    plt.show()

def propagate_signal_with_anomalies_in_place(x1, l1_holes, idx_anomaly, start_anomaly, intensity_anomaly, idx_rand_anomaly, start_rand, duration, time_instants):
    """
    Apply structured and random anomalies to the signal matrix.
    """

    print("idx_anomaly:", idx_anomaly)
    print("start_anomaly:", start_anomaly)
    print("idx_rand_anomaly:", idx_rand_anomaly)
    print("start_rand:", start_rand)

    intensity_rand = 1.001
    #x1[:, 0] = np.random.rand(NUM_EDGES)
    x1[:, 0] = np.ones(NUM_EDGES) * 0.5
    
    x0 = x1[:, 0]

    l1_holes_k = l1_holes.copy()

    for k in range(1, time_instants):
        # Propagate the signal

        if k > 1:
            l1_holes_k = np.dot(l1_holes_k, l1_holes)

        x1[:, k] = np.dot(l1_holes,x0)     #np.dot(l1_holes, x1[:, k-1])
        
        # Apply structured anomalies
        for i in range(len(idx_anomaly)):
            anomaly_index = idx_anomaly[i]
            start_time = start_anomaly[i]
            if k >= start_time:
                x1[anomaly_index, k] += x1[anomaly_index, k]*intensity_anomaly
        
        # Apply random anomalies
        #for j in range(len(idx_rand_anomaly)):
        #    rand_anomaly_index = idx_rand_anomaly[j]
        #    t0 = start_rand[j]
        #    t1 = t0 + duration
        #    if k >= t0 and k < t1:
        #        x1[rand_anomaly_index, k] += x1[rand_anomaly_index, k]*intensity_rand
    
    #plot last time step
    plt.figure(figsize=(10, 5))
    plt.plot(x1[:, -1], label='Signal at last time step', color='blue')
    plt.title('Signal at Last Time Step with Anomalies')
    plt.xlabel('Edge Index')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid()
    plt.show()

def optimization_loop_topology_known(x1, l1_holes, K):
    """
    Perform the optimization loop for all time steps.
    """

    lambda_val = 0.8
    u_Lnoto_all = np.zeros((NUM_EDGES, K-1))
    
    l1_holes_k = l1_holes.copy()

    x0 = x1[:, 0]  # Signal at time step 0

    for k in range(K-2):
        x_k = x1[:, k+1]  # Signal at time step k+1
        #x_km1 = x1[:, k]   # Signal at time step k
        
        if k > 0:
            l1_holes_k = np.dot(l1_holes_k, l1_holes)

        u_Lnoto = None
        u_Lnoto = cp.Variable(NUM_EDGES)

        objective = cp.Minimize(cp.norm(x_k - np.dot(l1_holes_k, x0) - u_Lnoto, 2)**2) #+ lambda_val * cp.norm(u_Lnoto, 1))
        
        prob = cp.Problem(objective)
        prob.solve(solver=cp.CLARABEL)  # SCS solver as an alternative to Sedumi
        
        u_Lnoto_all[:, k] = u_Lnoto.value

    return u_Lnoto_all

#TODO @tiziana help
def optimization_loop_topology_unknown(x1, l1_holes, K, num_cells_with_holes):
    """
    Perform the optimization loop for all time steps.
    """

    lambda_val = 0.5
    u_Lnoto_all = np.zeros((NUM_EDGES, K-1))
    
    for k in range(K-1):
        x_k = x1[:, k+1]  # Signal at time step k+1
        x_km1 = x1[:, k]   # Signal at time step k
        
        u_Lnoto = cp.Variable(NUM_EDGES)
        objective = cp.Minimize(cp.norm(x_k - np.dot(l1_holes, x_km1) + u_Lnoto, 'fro') + lambda_val * cp.norm(u_Lnoto, 1))
        
        prob = cp.Problem(objective)
        prob.solve(solver=cp.SCS)  # SCS solver as an alternative to Sedumi
        
        u_Lnoto_all[:, k] = u_Lnoto.value

    return u_Lnoto_all

def calculate_lambda_max(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return max(eigenvalues)#, eigenvectors

def main():
    # Generazione del grafo casuale
    graph, coords_matrix, adjacency_matrix = generate_random_graph(NUM_NODES, NUM_EDGES)

    while graph.number_of_edges() < NUM_EDGES or graph.number_of_nodes() < NUM_NODES:
        graph, coords_matrix, adjacency_matrix = generate_random_graph(NUM_NODES, NUM_EDGES)
    #plot_graph(graph, coords_matrix)

    print("Graph generated successfully.")
    B1, B2, selected_cells = func_gen_B2_lu(graph, 20)

    print("after func_gen_B2_lu")
    if PLOT_GRAPH:
        # Plot del cell complex
        fig, ax = plot_cell_complex(graph, coords_matrix, selected_cells,figsize=(8, 8), cmap='viridis',edge_color='k', node_size=30,node_facecolor='w', node_edgecolor='k',annotate=True, annot_fontsize=8)
        if SAVE_FIG:
            fig.savefig('cell_fig.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return

if __name__ == "__main__":
    print(cp.installed_solvers())
    main()