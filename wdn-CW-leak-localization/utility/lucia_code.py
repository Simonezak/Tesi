import networkx as nx

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