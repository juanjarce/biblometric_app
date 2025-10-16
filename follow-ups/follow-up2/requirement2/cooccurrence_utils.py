import itertools
import networkx as nx
from collections import defaultdict
import re

def build_cooccurrence_matrix(abstracts, terms):
    """
    Construye una matriz de coocurrencia entre términos completos (no subcadenas),
    considerando si aparecen juntos en un mismo abstract.
    """
    cooccurrence = defaultdict(int)
    term_set = [t.lower() for t, _ in terms]

    # Se compila regex para cada término (con límites de palabra)
    term_patterns = {t: re.compile(rf"\b{re.escape(t)}\b") for t in term_set}

    for abs_text in abstracts.values():
        text = abs_text.lower()
        found_terms = [t for t, pattern in term_patterns.items() if pattern.search(text)]

        # Combinaciones únicas no dirigidas
        for t1, t2 in itertools.combinations(sorted(found_terms), 2):
            cooccurrence[(t1, t2)] += 1

    return cooccurrence

def build_cooccurrence_graph(cooccurrence_dict, min_cooccurrence=1):
    """
    Construye un grafo no dirigido de coocurrencias.
    Cada nodo es un término y cada arista representa coocurrencia >= umbral.
    """
    G = nx.Graph()
    for (t1, t2), freq in cooccurrence_dict.items():
        if freq >= min_cooccurrence:
            G.add_edge(t1, t2, weight=freq)
    return G

def analyze_connected_components(G):
    """
    Analiza las componentes conexas del grafo de coocurrencia.
    Retorna un diccionario con:
      - total_components: número total de componentes
      - largest_component: conjunto de términos del componente más grande
      - all_components: lista de todos los conjuntos de términos
    """
    if G is None or len(G.nodes) == 0:
        print("No hay nodos en el grafo para analizar.")
        return {"total_components": 0, "largest_component": set(), "all_components": []}

    # obtiene componentes conexas (en grafos no dirigidos)
    components = list(nx.connected_components(G))
    total_components = len(components)
    print(f"\nSe encontraron {total_components} componentes conexas.")

    if not components:
        return {"total_components": 0, "largest_component": set(), "all_components": []}

    # identifica la componente más grande
    largest = max(components, key=len)
    print(f"Componente más grande: {len(largest)} términos -> {largest}")

    # muestra todas las componentes
    print("\nTemas detectados:")
    for i, comp in enumerate(components, 1):
        print(f"  - Tema {i} ({len(comp)} términos): {', '.join(sorted(comp))}")

    return {
        "total_components": total_components,
        "largest_component": largest,
        "all_components": components
    }