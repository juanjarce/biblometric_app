import networkx as nx
import matplotlib.pyplot as plt
import random

def visualize_graph_connected(
    G,
    output_path="follow-ups/follow-up2/outputs/citation_graph_connected.png",
    sample_size=60,
    label_type="id",
    show_edge_weights=True
):
    """
    Visualiza solo los nodos conectados entre sí (sin nodos aislados),
    mostrando también los pesos de las aristas si se desea.
    """
    print(f"Grafo original: {len(G)} nodos y {len(G.edges)} aristas")

    # Filtrar solo los nodos conectados
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    # Elige solo los componentes con más de 1 nodo
    connected_components = [c for c in components if len(c) > 1]

    if not connected_components:
        print("No hay componentes conectados con más de 1 nodo.")
        return

    # Toma el componente más grande
    largest_component = max(connected_components, key=len)
    G_connected = G.subgraph(largest_component).copy()
    print(f"Se muestra el componente más grande con {len(G_connected)} nodos y {len(G_connected.edges)} aristas")

    # Si sigue siendo muy grande, toma una muestra
    if len(G_connected) > sample_size:
        sampled_nodes = random.sample(list(G_connected.nodes()), sample_size)
        G_connected = G_connected.subgraph(sampled_nodes).copy()
        print(f"Muestra aleatoria de {len(G_connected)} nodos para visualización.")

    # Dibujar
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_connected, k=0.8, iterations=40, seed=42)
    nx.draw_networkx_edges(G_connected, pos, edge_color='gray', alpha=0.5, arrows=True)
    nx.draw_networkx_nodes(G_connected, pos, node_color='skyblue', node_size=400, alpha=0.9)

    # Etiquetas de nodos
    labels = {}
    for node in G_connected.nodes():
        if label_type == "title":
            title = G_connected.nodes[node].get("title", "")
            short_title = " ".join(title.split()[:4]) + ("..." if len(title.split()) > 4 else "")
            labels[node] = short_title
        else:
            labels[node] = node

    nx.draw_networkx_labels(G_connected, pos, labels=labels, font_size=7)

    # Etiquetas de aristas (pesos/similitud)
    if show_edge_weights:
        edge_labels = nx.get_edge_attributes(G_connected, 'weight')
        # Muestra con menos decimales
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G_connected, pos, edge_labels=edge_labels, font_size=6, alpha=0.7)

    plt.title(
        f"Grafo del componente más grande ({len(G_connected)} nodos conectados)",
        fontsize=12,
        fontweight='bold'
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafo conectado guardado en: {output_path}")