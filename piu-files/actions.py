# actions.py
"""
Definizione delle azioni possibili sugli elementi della rete WNTR.
Queste funzioni agiscono direttamente sul WaterNetworkModel (wn).
"""

from wntr.network.elements import LinkStatus


def open_pipe(wn, pipe_name: str):
    """Apre un tubo (imposta stato OPEN)."""
    pipe = wn.get_link(pipe_name)
    pipe.initial_status = LinkStatus.Open
    return wn


def close_pipe(wn, pipe_name: str):
    """Chiude un tubo (imposta stato CLOSED)."""
    pipe = wn.get_link(pipe_name)
    pipe.initial_status = LinkStatus.Closed
    return wn


def increase_pressure(wn, node_name: str, delta: float = 5.0):
    """
    Aumenta la pressione desiderata su un nodo.
    Implementato aumentando la domanda negativa (iniezione) o riducendo la domanda.
    """
    node = wn.get_node(node_name)
    try:
        node.base_demand = max(0.0, node.base_demand - delta)
    except AttributeError:
        pass  # se il nodo non ha base_demand
    return wn


def decrease_pressure(wn, node_name: str, delta: float = 5.0):
    """
    Diminuisce la pressione desiderata su un nodo.
    Implementato aumentando la domanda al nodo.
    """
    node = wn.get_node(node_name)
    try:
        node.base_demand = node.base_demand + delta
    except AttributeError:
        pass
    return wn


def noop(wn):
    """Non fare nulla (No Operation)."""
    return wn


def close_all_pipes(wn):
    """Chiudi tutti i tubi della rete."""
    for pipe_name in wn.pipe_name_list:
        wn.get_link(pipe_name).initial_status = LinkStatus.Closed
    return wn
