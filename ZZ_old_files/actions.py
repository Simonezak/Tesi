"""
Azioni per la rete idrica — compatibili con la branch Dynamic-WNTR
Usano direttamente i metodi di controllo di InteractiveWNTRSimulator.
"""

from wntr.network.elements import LinkStatus


def open_pipe(sim, pipe_name: str):
    """
    Apre un tubo specifico nella simulazione interattiva.
    """
    if not hasattr(sim, "open_pipe"):
        raise AttributeError("L'oggetto 'sim' non supporta open_pipe().")
    sim.open_pipe(pipe_name)
    print(f"[ACTION] Pipe '{pipe_name}' aperta.")


def close_pipe(sim, pipe_name: str):
    """
    Chiude un tubo specifico nella simulazione interattiva.
    """
    if not hasattr(sim, "close_pipe"):
        raise AttributeError("L'oggetto 'sim' non supporta close_pipe().")
    sim.close_pipe(pipe_name)
    print(f"[ACTION] Pipe '{pipe_name}' chiusa.")


def noop(sim):
    """
    Non esegue alcuna azione (step di mantenimento).
    """
    print("[ACTION] Nessuna azione (noop).")

# Per quest'azione bisogna vedere perche se chiudo tutti i tubi quando costruisco il grafo
# non ci sarà piu neanche un arco quindi va vista come implementarla
def close_all_pipes(sim):
    """
    Chiude tutti i tubi nella rete.
    """
    if not hasattr(sim, "close_pipe"):
        raise AttributeError("L'oggetto 'sim' non supporta close_pipe().")
    for pipe_name in sim._wn.pipe_name_list:
        sim.close_pipe(pipe_name)
    print("[ACTION] Tutti i tubi chiusi.")
