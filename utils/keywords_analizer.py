import bibtexparser
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rapidfuzz import fuzz
import os
import matplotlib.pyplot as plt

""""
Se definen las palabras clave 'keywords' bajo la categoría 'Concepts of Generative AI in Education'
para las cuales se van a calcular la frecuencia de aparición en los abstracts de los articulos finales
en 'meged.bib'
"""
keywords = [
    "generative models", "prompting", "machine learning", "multimodality",
    "fine-tuning", "training data", "algorithmic bias", "explainability",
    "transparency", "ethics", "privacy", "personalization",
    "human-ai interaction", "ai literacy", "co-creation"
]

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

""""
Funcion para cualcular la frecuencia de los keyword
en los abstracts de los articulos
"""
def count_keyword_frequencies(abstracts, keywords):
    counts = Counter()
    text = " ".join(abstracts.values()).lower()
    for kw in keywords:
        pattern = re.escape(kw.lower())
        matches = re.findall(pattern, text)
        counts[kw] = len(matches)
    return counts

""""
Función para obtener los 15 términos pricnipales dentro de los abstracts 
"""
def extract_top_terms(abstracts, top_n=15):
    # Convierte los abstracts (que vienen en un diccionario {id: texto}) en una lista de solo textos
    texts = list(abstracts.values())

    # Crea un vectorizador TF-IDF (Term Frequency – Inverse Document Frequency)
    #    - stop_words="english": ignora palabras muy comunes en inglés (the, and, of, etc.)
    #    - ngram_range=(1,3): considera unigramas (1 palabra), bigramas (2 palabras) y trigramas (3 palabras)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,3))

    tfidf = vec.fit_transform(texts) # Transforma los textos en una matriz TF-IDF

    scores = np.asarray(tfidf.sum(axis=0)).ravel()  # Calcula la suma de los pesos TF-IDF de cada término en todos los documentos

    terms = vec.get_feature_names_out() # Obtiene la lista de términos correspondientes a las columnas de la matriz TF-IDF

    # Empareja cada término con su puntuación y los ordena de mayor a menor
    ranking = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

    return ranking[:top_n]

""""
Función para calcular la precisión en el rango [-1, 1] de los nuevos términos
con los keywords iniciales de la categoría 'Concepts of Generative AI in Education'
"""
def precision_new_terms(new_terms, keywords, threshold=60):
    """
    Calcula la precisión de coincidencias entre top términos (TF-IDF)
    y los keywords definidos.
    Se usa fuzzy matching con un umbral de similitud (default=60/100).
    """
    new_set = set([t for t, _ in new_terms])
    keyword_set = set(k.lower() for k in keywords)

    overlap = set()
    for kw in keyword_set:
        for term in new_set:
            score = fuzz.partial_ratio(term, kw)
            if score >= threshold:
                overlap.add(kw)

    return len(overlap) / len(new_set), overlap

## --------------------------------------------------------------------------------------------------------------------------------
# Gráficas de visualización de resultados

""""
Función para generar la gráfica de frecuencia de los keywords 'Concepts of Generative AI in Education'
en los abstracts de los articulos
"""
def plot_keyword_frequencies(freqs, output_path="outputs/keywords_analizer/keywords_freq.png"):
    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ordenar keywords por frecuencia descendente
    sorted_items = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    keywords, counts = zip(*sorted_items)

    # Gráfica de barras
    plt.figure(figsize=(10, 6))
    plt.bar(keywords, counts, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Keywords")
    plt.ylabel("Frecuencia")
    plt.title("Frecuencia de Keywords en Abstracts")
    plt.tight_layout()

    # Guardar en archivo
    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Gráfica guardada en {output_path}")

""""
Función para generar la gráfica con los nuevos keywords
los nuevos keywords son los 15 terminos principales en los abstracts de los articulos
en los abstracts de los articulos.
Para cada keywords nuevo se muestran 2 tipos de información clave:
- Frecuencia de aparición en los abstracts
- TF-IDF: qué tan representativa es la palabra o término dentro de el conjunto de documentos (abstracts)
"""
def plot_new_terms_tfidf_vs_freq(abstracts, top_terms, output_path="outputs/keywords_analizer/new_terms_tfidf_freq.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calcular frecuencias de cada termino en los abstracts
    text = " ".join(abstracts.values()).lower()
    freqs = {}
    for term, score in top_terms:
        pattern = re.escape(term.lower())
        matches = re.findall(pattern, text)
        freqs[term] = len(matches)

    # Extraer los datos para las graficas
    terms = [t for t, _ in top_terms]
    tfidf_scores = [s for _, s in top_terms]
    counts = [freqs[t] for t in terms]

    # Gráfico combinado
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Barras para frecuencia
    ax1.bar(terms, counts, color="skyblue", alpha=0.7, label="Frecuencia")
    ax1.set_ylabel("Frecuencia", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticklabels(terms, rotation=45, ha="right")

    # Eje secundario para TF-IDF
    ax2 = ax1.twinx()
    ax2.plot(terms, tfidf_scores, color="red", marker="o", linewidth=2, label="TF-IDF")
    ax2.set_ylabel("TF-IDF Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Título y layout
    plt.title("Top 15 términos: TF-IDF vs Frecuencia")
    fig.tight_layout()

    # Guardar
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Gráfica combinada guardada en {output_path}")

"""
Función para visualizar la precisión de los nuevos keywords con los keywords originales:
Determina qué tan precisas son las nuevas palabras.
"""
def plot_precision_and_terms(precision, common_terms, output_path="outputs/keywords_analizer/precision_terms.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    # --- 1. Gauge chart para precisión ---
    ax = axes[0]
    ax.axis("equal")
    # fondo gris
    ax.pie([1], radius=1, colors=["#E0E0E0"], wedgeprops=dict(width=0.3, edgecolor="white"))
    # precisión (verde hasta el valor)
    ax.pie([precision, 1-precision], radius=1, startangle=90, colors=["#4CAF50", "none"],
           wedgeprops=dict(width=0.3, edgecolor="white"))
    # texto central
    ax.text(0,0,f"{precision*100:.0f}%", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title("Precisión", fontsize=14)

    # --- 2. Barras para términos en común ---
    ax2 = axes[1]
    terms = list(common_terms)
    counts = [1]*len(terms)  # cada término encontrado cuenta como 1
    ax2.barh(terms, counts, color="#2196F3")
    ax2.set_xlim(0,1.5)
    ax2.set_title("Términos en común", fontsize=14)
    ax2.set_xlabel("Encontrado (1=Sí)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Gráfica guardada en {output_path}")

## --------------------------------------------------------------------------------------------------------------------------------
# main
def main_keywords_analizer():
        # Se cargan los abstracts del 'merged.bib'
    abstracts = load_bib()

    # --- Frecuencias de keywords ---
    freqs = count_keyword_frequencies(abstracts, keywords)
    # Generar gráfica
    plot_keyword_frequencies(freqs)
    """"
    print("\n=== Frecuencia de keywords ===")
    for kw, c in freqs.items(): 
        print(f"{kw}: {c}")
    """

    # --- Términos principales (TF-IDF) ---
    top_terms = extract_top_terms(abstracts, top_n=15)
    # Generar gráfica
    plot_new_terms_tfidf_vs_freq(abstracts, top_terms)
    """
    print("\n=== Top términos por TF-IDF ===")
    for term, score in top_terms:
        print(f"{term}: {score:.4f}")
    """

    # --- Precisión de nuevos términos vs keywords ---
    precision, overlap = precision_new_terms(top_terms, keywords)
    # Generar gráfica
    plot_precision_and_terms(precision, overlap)
    """"
    print("\n=== Precisión de nuevos términos ===")
    print(f"Precisión: {precision:.2f}")
    print(f"Términos en común: {overlap}")
    """

if __name__ == "__main__":
    main_keywords_analizer()