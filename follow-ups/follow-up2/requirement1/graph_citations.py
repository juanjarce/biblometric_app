import networkx as nx
from itertools import combinations

def similarity(article1, article2):
    """Calcula similitud bibliométrica basada en autores, palabras clave y título."""
    authors1, authors2 = set(a.lower() for a in article1["authors"]), set(a.lower() for a in article2["authors"])
    keywords1, keywords2 = set(k.lower() for k in article1["keywords"]), set(k.lower() for k in article2["keywords"])
    
    # Similitud por intersección proporcional
    author_sim = len(authors1 & authors2) / max(1, len(authors1 | authors2))
    keyword_sim = len(keywords1 & keywords2) / max(1, len(keywords1 | keywords2))
    
    # Similitud de título usando palabras comunes
    words1 = set(article1["title"].lower().split())
    words2 = set(article2["title"].lower().split())
    common_words = len(words1 & words2)
    title_sim = common_words / max(1, len(words1 | words2))
    
    # Ponderación
    score = (0.5 * author_sim) + (0.3 * keyword_sim) + (0.2 * title_sim)
    return round(score, 3)

def build_citation_graph(articles, threshold=0.4):
    """
    Construye el grafo de citaciones dirigido con pesos según similitud.
    Para que se considere que un articulo cita a otro debe de estar con una similitud mayor al umbral (threshold=0.4)
    """
    G = nx.DiGraph()
    for art in articles:
        G.add_node(art["id"], title=art["title"], year=art["year"])

    for a1, a2 in combinations(articles, 2):
        sim = similarity(a1, a2)
        if sim >= threshold:
            G.add_edge(a1["id"], a2["id"], weight=sim)
    return G