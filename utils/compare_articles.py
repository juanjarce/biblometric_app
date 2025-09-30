import bibtexparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textdistance
from sentence_transformers import SentenceTransformer
import argparse

""""
Ejemplo de entrada:
python3 utils/compare_articles.py --compare_ids merged1,merged3
"""

""""
Función para obtener la etiqueta de los articulos con sus respectivos abstracts
se traen en el formato:
    {
        "merged1": "texto del abstract 1",
        "merged2": "texto del abstract 2",
        ...
    }
"""
def load_bib(path="data/processed/merged.bib"):
    with open(path, encoding="utf-8") as f:
        db = bibtexparser.load(f)
        return {entry["ID"]: entry.get("abstract", "") for entry in db.entries}
    
def levenshtein_sim(s1, s2):
    return 1 - textdistance.levenshtein.normalized_distance(s1, s2)

def jaccard_sim(s1, s2):
    set1, set2 = set(s1.lower().split()), set(s2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def dice_sim(s1, s2):
    set1, set2 = set(s1.lower().split()), set(s2.lower().split())
    return 2 * len(set1 & set2) / (len(set1) + len(set2))

def tfidf_cosine(s1, s2): 
    vec = TfidfVectorizer().fit([s1, s2])
    tfidf = vec.transform([s1, s2])
    return cosine_similarity(tfidf[0], tfidf[1])[0][0]

def sbert_cosine(s1, s2):
    """"
    - Modelo más pequeño y rápido, con menos parámetros.
    - Ideal si priorizas velocidad sobre precisión.
    - Recomendado para aplicaciones en tiempo real o con muchos textos.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2") # Carga un modelo SBERT ligero
    # Genera (vectores) para los abstracts s1 y s2
    emb = model.encode([s1, s2])
    # Calcula la similitud coseno
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def compare_articles_sbert(s1, s2):
    """"
    - Modelo más grande y preciso, entrenado para similitud semántica.
    - Captura relaciones semánticas profundas.
    - Segun lo investigado se utiliza para tareas de semantic textual similarity (STS).
    """
    # Usamos el mismo modelo potente
    sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Generamos (vectores) de los dos textos
    embeddings = sbert_model.encode([s1, s2], convert_to_numpy=True)
    # Calculamos la similitud coseno entre los abstracts
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim

""""
Función para comparar (todos con todos) los articulos seleccionados para su comparación
La comparación se hace con todas la funciones requeridas
    :param abstracts: diccionario con abstracts, ej. {"merged1": "...", "merged2": "..."}
    :param article_ids: lista de IDs a comparar, ej. ["merged1", "merged2"]
"""
def compare_articles(abstracts, ids):
    n = len(ids)
    for i in range(n):
        for j in range(i+1, n):
            s1, s2 = abstracts[ids[i]], abstracts[ids[j]]
            print(f"\nComparando {ids[i]} vs {ids[j]}:")
            print(" Levenshtein:", levenshtein_sim(s1, s2))
            print(" Jaccard:", jaccard_sim(s1, s2))
            print(" Dice:", dice_sim(s1, s2))
            print(" TF-IDF Cosine:", tfidf_cosine(s1, s2))
            print(" SBERT Cosine 1:", sbert_cosine(s1, s2))
            print(" SBERT Cosine 2:", compare_articles_sbert(s1, s2))
""""
Misma función para comparar los abstracts de los articulos indicados
Modificada para el uso de 'Streamlit' en el menu web
"""
def run_comparison_from_ids(ids_str):
    abstracts = load_bib()
    ids = ids_str.split(",")
    output = []

    n = len(ids)
    for i in range(n):
        for j in range(i+1, n):
            s1, s2 = abstracts[ids[i]], abstracts[ids[j]]
            output.append(f"\nComparando {ids[i]} vs {ids[j]}:")
            output.append(f" Levenshtein: {levenshtein_sim(s1, s2)}")
            output.append(f" Jaccard: {jaccard_sim(s1, s2)}")
            output.append(f" Dice: {dice_sim(s1, s2)}")
            output.append(f" TF-IDF Cosine: {tfidf_cosine(s1, s2)}")
            output.append(f" SBERT Cosine 1: {sbert_cosine(s1, s2)}")
            output.append(f" SBERT Cosine 2: {compare_articles_sbert(s1, s2)}")

    return "\n".join(output)
    
# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta de comparación de artículos")
    parser.add_argument("--compare_ids", type=str, help="IDs de artículos a comparar, separados por coma. Ej: --compare_ids merged1,merged2")
    args = parser.parse_args()

    if args.compare_ids:
        output = run_comparison_from_ids(args.compare_ids)
        print(output)