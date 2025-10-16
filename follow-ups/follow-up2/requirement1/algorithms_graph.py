import networkx as nx

def shortest_path_dijkstra(G, source, target):
    """Encuentra el camino más corto entre dos nodos usando Dijkstra."""
    try:
        path = nx.dijkstra_path(G, source, target, weight="weight")
        dist = nx.dijkstra_path_length(G, source, target, weight="weight")
        return path, dist
    except nx.NetworkXNoPath:
        return None, float("inf")

def all_pairs_shortest_paths(G):
    """Matriz de caminos mínimos (Floyd–Warshall)."""
    return dict(nx.floyd_warshall(G, weight="weight"))

def strongly_connected_components(G):
    """Encuentra las componentes fuertemente conexas."""
    return list(nx.strongly_connected_components(G))
