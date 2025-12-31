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
 
