# Se añade a raíz del proyecto para poder usar el utils
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from requirement1.parser_bib import parse_bib_file
from requirement1.graph_citations import build_citation_graph
from requirement1.algorithms_graph import shortest_path_dijkstra, strongly_connected_components
from requirement1.visualize_graph import visualize_graph_connected

from utils.keywords_analizer import extract_top_terms
from requirement2.parser_bib import load_abstracts_from_bib
from requirement2.cooccurrence_utils import build_cooccurrence_matrix, build_cooccurrence_graph, analyze_connected_components
from requirement2.visualize_cooccurrence_graph import visualize_cooccurrence_graph

def main_requirement1():
    print(f"\n---------------- Requerimiento 1 ----------------")

    # 1. Cargar artículos
    articles = parse_bib_file("data/processed/merged.bib")
    print(f"Se cargaron {len(articles)} artículos.")

    # 2. Construir grafo
    G = build_citation_graph(articles, threshold=0.4)
    print(f"Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

    # 3. Calcular caminos mínimos
    source_id = "merged189"
    target_id = "merged260"

    if source_id in G.nodes() and target_id in G.nodes():
        path, dist = shortest_path_dijkstra(G, source_id, target_id)
        if path is not None:
            print(f"Camino más corto entre {source_id} y {target_id}: {path} (distancia: {dist})")
        else:
            print(f"No existe camino entre {source_id} y {target_id}.")
    else:
        print("Uno o ambos nodos no existen en el grafo.")

    # 4. Componentes fuertemente conexas
    components = strongly_connected_components(G)
    print(f"Componentes fuertemente conexas encontradas: {len(components)}")

    # 5. Visualizar grafo
    visualize_graph_connected(G, "follow-ups/follow-up2/outputs/citation_graph_connected.png", label_type="id", show_edge_weights=True)

def main_requirement2():
    print(f"\n---------------- Requerimiento 2 ----------------")

    # 1. Cargar abstracts
    abstracts = load_abstracts_from_bib("data/processed/merged.bib")
    print(f"Se cargaron {len(abstracts)} abstracts.")

    # 2. Extraer los 15 términos principales
    top_terms = extract_top_terms(abstracts, top_n=15)
    print("Términos principales extraídos:")
    for term, score in top_terms:
        print(f" - {term}: {score:.4f}")

    # 3. Construir matriz y grafo
    cooc_matrix = build_cooccurrence_matrix(abstracts, top_terms)
    G = build_cooccurrence_graph(cooc_matrix, min_cooccurrence=1)
    print(f"\nGrafo de coocurrencia creado con {len(G.nodes())} nodos y {len(G.edges())} aristas.")

    # 4. Detección de componentes conexas
    component_info = analyze_connected_components(G)

    # 5. Visualización mejorada
    print("\n")
    visualize_cooccurrence_graph(G)

if __name__ == "__main__":
    main_requirement1()
    main_requirement2()
