import matplotlib.pyplot as plt
from wntr.network.elements import LinkStatus
import networkx as nx
import numpy as np
from main_dyn_topologyknown_01 import func_gen_B2_lu, plot_cell_complex
from matplotlib import colors

def get_inital_polygons_flux_limits(f_polygons):
    """
    Calcola vmin e vmax dai valori assoluti dei flussi poligonali.

    Args:
        f_polygons : np.ndarray
            Vettore o matrice [Npolygons x 1] dei flussi.
        vmin, vmax : float o None
            Valori opzionali. Se forniti, vengono mantenuti.

    Returns:
        (vmin, vmax) : tuple[float, float]
    """

    flux_abs = np.abs(f_polygons.flatten())
    
    vmin = float(flux_abs.min())
    vmax = float(flux_abs.max())

    return vmin, vmax


def construct_matrix_f(wn, results):
    """
    Costruisce la matrice colonna dei flussi (f) dai risultati WNTR, considerando solo i tubi Junction (no reservoir e tank)
    """
    df_flow = results.link["flowrate"].iloc[-1]  # flusso all’ultimo timestep

    f = np.array(df_flow, dtype=float).reshape(-1, 1)

    return f


    
def plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin, vmax,
                      leak_node=None, step=None,
                      figsize=(6,6),
                      cmap='plasma',
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
      f_polygons      (np.ndarray): [Ncycles x 1] flussi netti per ciclo
      vmin, vmax      (float): limiti min e max per la colormap
      leak_node_name  (str, opzionale): nome del nodo con perdita
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

    # --- Normalizzazione e colormap ---
    flux_abs = np.abs(f_polygons.flatten())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    # 1) cycle fills
    # a random color value per cycle, normalized 0..1

    for ci, cycle in enumerate(selected_cycles):
        pts = coords[cycle]
        poly = np.vstack([pts, pts[0]])
        color = cmap_obj(norm(flux_abs[ci])) if ci < len(flux_abs) else "gray"
        ax.fill(poly[:, 0], poly[:, 1],
                facecolor=color, edgecolor="none", alpha=0.6)

        if annotate:
            centroid_x = np.mean(poly[:, 0])
            centroid_y = np.mean(poly[:, 1])
            flux_val = f_polygons[ci, 0]
            ax.text(centroid_x, centroid_y, f"{flux_val:.3f}",
                    color="black", fontsize=annot_fontsize,
                    ha="center", va="center",
                    fontweight="bold", zorder=12)    


    # archi e nodi base
    for u, v in G.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.plot([x0, x1], [y0, y1], color="k", linewidth=1)

    ax.scatter(coords[:, 0], coords[:, 1], s=30, facecolor="w", edgecolor="k", zorder=3)

    # --- Archi ---
    for u, v in G.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1)

    # --- Nodi ---
    ax.scatter(coords[:, 0], coords[:, 1],
               s=node_size,
               facecolor=node_facecolor,
               edgecolor=node_edgecolor,
               zorder=3)
            
    # Nodo leak
    if leak_node is not None:
        x, y = leak_node.coordinates
        ax.scatter(x, y, color="red", s=node_size * 1.5, marker="o",
                   edgecolor="black", linewidths=1.0, zorder=10)
        ax.text(x + 2, y + 2, "LEAK",
                color="red", fontsize=10, fontweight="bold", zorder=11)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    fig.colorbar(sm, ax=ax, label="Flusso netto per poligono [m³/s]")

    # --- Titolo e stile ---
    ax.set_title(f"Flowrate poligoni - step {step}" if step is not None else "Flowrate poligoni")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig, ax


def compute_polygon_flux(f, B2):
    """
    Calcola il flusso netto per ciascun poligono
    in base al vettore dei flussi (f) e alla matrice topologica B2.

    f:  [Nedge x 1]  vettore dei flussi (in m³/s)
    B2: [Nedge x Npolygons] matrice topologica (da func_gen_B2_lu)

    Ritorna:
        f_polygons: [Npolygons x 1] vettore dei flussi per poligono
    """

    # Moltiplicazione Matrici B2' * f
    f_polygons = B2.T @ f

    return f_polygons
