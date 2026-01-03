import matplotlib.pyplot as plt
from wntr.network.elements import LinkStatus
import networkx as nx
import numpy as np
from main_dyn_topologyknown_01 import func_gen_B2_lu, plot_cell_complex
from matplotlib import colors

import numpy as np

def dense_rank_descending(scores):
    """
    Restituisce il dense-rank (0-based) per ogni elemento.
    Score più alto = rank 0.
    Score uguali = stesso rank.

    Esempio:
    scores = [10, 9, 9, 7]
    ranks  = [0, 1, 1, 2]
    """
    scores = np.asarray(scores)

    # valori unici ordinati in modo decrescente
    unique_scores = np.unique(scores)[::-1]

    score_to_rank = {s: i for i, s in enumerate(unique_scores)}

    ranks = np.array([score_to_rank[s] for s in scores])
    return ranks

def leak_dense_positions(score_per_node, idx2node, leak_nodes):
    """
    Restituisce le posizioni dense-rank (0-based) dei leak reali.
    """
    ranks = dense_rank_descending(score_per_node)

    positions = []
    for ln in leak_nodes:
        leak_idx = [i for i, n in idx2node.items() if n == ln][0]
        positions.append(ranks[leak_idx])

    return positions


def evaluate_single_test_lexicographic(
    score_per_node,
    idx2node,
    leak_nodes,
    X=2
):
    """
    Valuta un singolo test usando ranking lessicografico (dense).

    Metriche:
    - topX
    - top10
    - topX_single
    - top10_single
    """

    pos = leak_dense_positions(score_per_node, idx2node, leak_nodes)

    if len(pos) == 0:
        return {
            "topX": False,
            "top10": False,
            "topX_single": False,
            "top10_single": False
        }

    pos = sorted(pos)

    # 🔹 Tutti i leak contigui e nelle prime X posizioni
    topX = (
        max(pos) < X and
        max(pos) - min(pos) == len(pos) - 1
    )

    # 🔹 Tutti i leak entro top 10
    top10 = max(pos) < 10

    # 🔹 Almeno un leak in prima posizione
    topX_single = min(pos) == 0

    # 🔹 Almeno un leak entro top 10
    top10_single = min(pos) < 10

    return {
        "topX": topX,
        "top10": top10,
        "topX_single": topX_single,
        "top10_single": top10_single
    }

def evaluate_model_across_tests_lexicographic(
    scores_per_test,
    idx2node,
    leak_nodes_per_test,
    X=2
):
    """
    Valuta un modello su tutti i test usando ranking lessicografico.

    scores_per_test     : list[np.ndarray]  -> score_per_node per test
    leak_nodes_per_test : list[list[str]]   -> leak reali per test
    """

    counters = {
        "topX": 0,
        "top10": 0,
        "topX_single": 0,
        "top10_single": 0
    }

    num_tests = len(scores_per_test)

    for score_per_node, leak_nodes in zip(scores_per_test, leak_nodes_per_test):

        res = evaluate_single_test_lexicographic(
            score_per_node=score_per_node,
            idx2node=idx2node,
            leak_nodes=leak_nodes,
            X=X
        )

        for k in counters:
            counters[k] += int(res[k])

    # conversione in percentuali
    for k in counters:
        counters[k] = 100.0 * counters[k] / num_tests

    return counters 


# ============================================================
#  DA AGGIORNARE SE DA TENERE
# ============================================================

def plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin, vmax,
                      leak_node=None, episode=None, step=None, test=False,
                      figsize=(8,8),
                      cmap='plasma',
                      edge_color='k',
                      node_size=40,
                      node_facecolor='w',
                      node_edgecolor='k',
                      annotate=True,
                      annot_fontsize=9):
    """
    Plot the graph G with its selected cycle‐cells overlaid.
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
    ax.set_title(f"Flowrate poligoni - Episodio {episode}, step {step}" if step is not None else "Flowrate poligoni")
    if test:
        ax.set_title(f"TEST WN - Flowrate poligoni")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig, ax


def visualize_snapshot(all_snapshots, episode_id, step, wn, results):
    """
    Visualizza lo stato WN corrispondente a uno snapshot specifico
    """

    # Trova lo snapshot corrispondente
    snap = next(
        (d for d in all_snapshots
         if getattr(d, "episode_id", None) == episode_id
         and getattr(d, "step", None) == step),
        None
    )
    if snap is None:
        print(f"[ERRORE] Nessuno snapshot trovato per episodio={episode_id}, step={step}")
        return

    # Ricostruisci grafo e calcola grandezze topologiche
    G, coords = build_nx_graph_from_wntr(wn, results)
    B1, B2, selected_cycles = func_gen_B2_lu(G, max_cycle_length=8)

    f = construct_matrix_f(wn, results)
    f_polygons = compute_polygon_flux(f, B2, abs=False)
    f_polygons_abs = compute_polygon_flux(f, B2, abs=True)

    # Limiti per le scale colore
    vmin_p, vmax_p = get_inital_polygons_flux_limits(f_polygons)
    vmin_n, vmax_n = get_initial_node_demand_limits(G)
    vmin_e, vmax_e = get_initial_edge_flow_limits(f)

    # Individua il nodo di leak dallo snapshot (etichetta y=1)
    leak_idx = (snap.y.squeeze() == 1).nonzero(as_tuple=True)[0]
    leak_node = None
    if len(leak_idx) > 0:
        leak_node_name = list(G.nodes())[int(leak_idx[0])]
        leak_node = wn.get_node(leak_node_name)
        print(f"[INFO] Leak al nodo: {leak_node_name}")

    print(f"\n Visualizzazione episodio={episode_id}, step={step}")
    plot_node_demand(G, coords, vmin_n, vmax_n, episode=episode_id, step=step)
    plot_edge_flowrate(G, coords, f, vmin_e, vmax_e, episode=episode_id, step=step)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons, vmin_p, vmax_p, leak_node, episode=episode_id, step=step)
    plot_cell_complex_flux(G, coords, selected_cycles, f_polygons_abs, vmin_p, vmax_p, leak_node, episode=episode_id, step=step)



def plot_node_demand(G, coords, vmin, vmax, figsize=(8,8), cmap='coolwarm', node_size=60,
                     edge_color='k', node_edgecolor='none', annotate=True, annot_fontsize=8, episode=None,
                     step=None, test=False):
    """
    Visualizza il grafo con i nodi colorati in base alla demand.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ottieni i valori di demand dai nodi
    junction_nodes = [n for n in G.nodes() if G.nodes[n].get("type", "") == "Junction"]
    demands = np.array([G.nodes[n].get('demand', "ND") if n in junction_nodes else 0.0 for n in G.nodes()])
    norm = colors.Normalize(vmin, vmax)
    cmap_obj = plt.get_cmap(cmap)

    # Disegna archi
    for u, v in G.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1, zorder=1)

    # Disegna nodi colorati
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    s=node_size * 2,
                    c=demands,
                    cmap=cmap_obj,
                    norm=norm,
                    edgecolor=node_edgecolor,
                    zorder=3)

    # Annotazioni
    if annotate:
        for i, (x, y) in enumerate(coords):
            if i in junction_nodes:
                val = demands[i]
                ax.text(x, y, f"{val:.3f}",
                        color="black",
                        fontsize=annot_fontsize,
                        ha="center", va="center",
                        fontweight="bold", zorder=5)




    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    fig.colorbar(sm, ax=ax, label="Demand")

    # Titolo e stile
    ax.set_title(f"Nodes Demand - Episodio {episode}, step {step}" if step else "Nodes Demand")
    if test:
        ax.set_title(f"TEST WN - Nodes Demand")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig, ax



def plot_edge_flowrate(G, coords, f, vmin, vmax,
                       figsize=(8,8),
                       cmap='coolwarm',
                       node_size=40,
                       annotate=True,
                       annot_fontsize=8, episode=None,
                       step=None, test=False):
    """
    Visualizza il grafo con gli archi colorati in base al flowrate normalizzato.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalizzazione
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    edge_colors = [cmap_obj(norm(val)) for val in f]

    # Disegna grafo
    nx.draw_networkx_edges(G, pos=dict(enumerate(coords)),
                           edge_color=edge_colors,
                           width=2.5,
                           ax=ax)
    
    # Disegna nodi sopra
    nx.draw_networkx_nodes(G, pos=dict(enumerate(coords)),
                           node_size=node_size,
                           node_color='white',
                           edgecolors='black',
                           ax=ax)

    # Annotazioni: valore del flowrate al centro di ciascun arco
    if annotate:
        for (i, (u, v)) in enumerate(G.edges()):
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"{float(f[i]):.3f}",
                    fontsize=annot_fontsize,
                    color='black',
                    ha='center', va='center',
                    fontweight='bold', zorder=5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    fig.colorbar(sm, ax=ax, label="Flowrate")

    # Titolo e stile
    ax.set_title(f"Flowrate negli archi - Episodio {episode}, step {step}" if step else "Flowrate negli archi",
                 fontsize=11)
    if test:
        ax.set_title(f"TEST WN - Flowrate archi")
    
    
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig, ax


def compute_polygon_flux(f, B2, abs: bool = False):
    """
    Calcola il flusso netto per ciascun poligono
    in base al vettore dei flussi (f) e alla matrice topologica B2.

    f:  [Nedge x 1]  vettore dei flussi (in m³/s)
    B2: [Nedge x Npolygons] matrice topologica (da func_gen_B2_lu)

    Ritorna:
        f_polygons: [Npolygons x 1] vettore dei flussi per poligono
    """

    if abs:
        # Flusso "non orientato": somma dei moduli per ogni poligono
        f = np.abs(f)
        B2 = np.abs(B2)

    # Moltiplicazione Matrici B2' * f
    f_polygons = B2.T @ f

    return f_polygons


def plot_edge_Uhat(G, coords, U_hat, vmin=None, vmax=None, cmap="coolwarm", step=None):
    """
    Plotta il valore U_hat sugli archi per un singolo step (stile identico a plot_edge_flowrate).
    Mostra:
      - nodi come punti neri
      - archi colorati in base al valore di U_hat
      - testo con il valore numerico al centro dell'arco
    """
    # --- Conversione a NumPy
    U_np = U_hat.detach().cpu().numpy().flatten() if hasattr(U_hat, "detach") else np.array(U_hat).flatten()
    edges = list(G.edges())

    # --- Range colori automatico
    if vmin is None: vmin = np.min(U_np)
    if vmax is None: vmax = np.max(U_np)

    norm = plt.Normalize(vmin, vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    plt.figure(figsize=(7,6))
    ax = plt.gca()

    # --- Disegna archi colorati
    for i, (u, v) in enumerate(edges):
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        color = cmap_obj(norm(U_np[i]))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, zorder=1)

        # testo del valore numerico (al centro del tubo)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, f"{U_np[i]:.3f}", color="black", fontsize=8,
                ha="center", va="center", zorder=3)

    # --- Nodi come punti neri
    nx.draw_networkx_nodes(G, coords, node_size=40, node_color="black", ax=ax)

    # --- Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("U_hat (anomalia flusso)")

    title = f"U_hat per step {step} - {step + 1}" if step is not None else "U_hat (anomalia flusso)"
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_edge_s_u(G, coords, s_u, vmin=None, vmax=None, cmap="plasma", leak_node=None,
                  node_size=30, node_facecolor="w", node_edgecolor="k"):
    """
    Plotta la mappa finale s_u(i) = sum_k |U_i[k]| su tutti gli archi,
    con struttura del grafo formattata come plot_cell_complex_flux:
      - archi neri di base
      - nodi bianchi con bordo nero
      - valori numerici centrali sugli archi
      - nodo del leak rosso con etichetta "LEAK"
    """

    # --- Conversione robusta a NumPy
    if hasattr(s_u, "detach"):
        s_u_np = s_u.detach().cpu().numpy()
    elif isinstance(s_u, (list, tuple)):
        s_u_np = np.array(s_u)
    else:
        s_u_np = np.array(s_u)
    s_u_np = s_u_np.flatten()
    edges = list(G.edges())

    # --- Range colori
    if vmin is None:
        vmin = float(np.min(s_u_np))
    if vmax is None:
        vmax = float(np.max(s_u_np))

    norm = plt.Normalize(vmin, vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    # --- Crea figura
    fig, ax = plt.subplots(figsize=(8, 7))

    # --- Disegna grafo base (archi neri sottili + nodi bianchi)
    for u, v in G.edges():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.plot([x0, x1], [y0, y1], color="k", linewidth=1, zorder=1)

    node_xy = np.array([coords[n] for n in G.nodes()])
    ax.scatter(node_xy[:, 0], node_xy[:, 1],
               s=node_size,
               facecolor=node_facecolor,
               edgecolor=node_edgecolor,
               zorder=3)

    # --- Disegna archi colorati in base a s_u
    for i, (u, v) in enumerate(edges):
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        color = cmap_obj(norm(s_u_np[i]))
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=3, zorder=2)

        # testo numerico al centro dell'arco
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        ax.text(mid_x, mid_y, f"{s_u_np[i]:.3f}", color="black", fontsize=8,
                ha="center", va="center", zorder=5)


    if leak_node is not None:
        x, y = leak_node.coordinates

        ax.scatter(x, y, color="red", s=node_size * 1.5, marker="o",
                    edgecolor="black", linewidths=1.0, zorder=10)
        ax.text(x + 2, y + 2, "LEAK", color="red",
                fontsize=10, fontweight="bold", zorder=11)

    # --- Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$s_u = \sum_k |U[k]|$")

    ax.set_title("Mappa finale s_u (intensità anomalia)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

