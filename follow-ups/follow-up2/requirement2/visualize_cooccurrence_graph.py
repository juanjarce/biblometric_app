import os
import matplotlib.pyplot as plt
import networkx as nx

def visualize_cooccurrence_graph(
    G,
    output_path="follow-ups/follow-up2/outputs/cooccurrence_graph.png"
):
    """
    Visualiza el grafo de coocurrencia con estilo similar al establecido en el proyecto.
    Todos los nodos tienen el mismo color, pero muestran su grado dentro del nodo (por ejemplo, '14°').
    """
    if G is None or len(G.nodes) == 0:
        print("No hay nodos en el grafo para visualizar.")
        return

    print(f"Grafo de coocurrencia: {len(G.nodes)} nodos y {len(G.edges)} aristas")

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Layout del grafo
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=50)

    # Tamaño del nodo según su grado
    node_sizes = [300 + G.degree(n) * 200 for n in G.nodes()]

    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.3, width=0.8)

    # Dibujar nodos (todos del mismo color)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="skyblue",
        edgecolors="gray",
        linewidths=1.0,
        alpha=0.95
    )

    # Etiquetas: nombre + grado con símbolo °
    labels = {n: f"{n}\n{G.degree(n)}°" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")

    # Título y formato
    plt.title(
        "Grafo de coocurrencia de términos\n"
        "(Tamaño = grado, etiqueta = grado°)",
        fontsize=14,
        fontweight="bold"
    )
    plt.axis("off")
    plt.tight_layout()

    # Guardar
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafo de coocurrencia guardado en: {output_path}")
