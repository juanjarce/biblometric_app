import bibtexparser
import time
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter
# Importar todos los 12 métodos de prdenamiento aplicados en 'sorting_algorithms.py'
from sorting_algorithms_adapted_key import tim_sort, comb_sort, selection_sort, tree_sort, pigeonhole_sort, bucket_sort, quick_sort, heap_sort, bitonic_sort, gnome_sort, binary_insertion_sort, radix_sort

"""
Función para cargar la información de los articulos del archivo unificado
Se trae solo la información necesaria para el ordenamiento según lo criterios:
    {
        "id": entry.get("ID", ""),
        "title": entry.get("title", "").strip(),
        "year": int(entry.get("year", 0)),
        "type": entry.get("ENTRYTYPE", ""),
    }
La ruta del archivo de info unificado es 'data/proccessed/merged.bib'
"""
def load_products(path="data/processed/merged.bib"):
    with open(path, encoding="utf-8") as f:
        db = bibtexparser.load(f)
        productos = []
        for entry in db.entries:
            productos.append({
                "id": entry.get("ID", ""),
                "title": entry.get("title", "").strip(),
                "year": int(entry.get("year", 0)),
                "type": entry.get("ENTRYTYPE", ""),
            })
        return productos

# -------------------------------------------------------------------------------------------------------------------------------- #
# Se organizan los métodos de ordenamiento como diccionario

algorithms = {
    "TimSort": tim_sort,
    "Comb Sort": comb_sort,
    "Selection Sort": selection_sort,
    "Tree Sort": tree_sort,
    "Pigeonhole Sort": pigeonhole_sort,
    "Bucket Sort": bucket_sort,
    "QuickSort": quick_sort,
    "HeapSort": heap_sort,
    "Bitonic Sort": bitonic_sort,
    "Gnome Sort": gnome_sort,
    "Binary Insertion Sort": binary_insertion_sort,
    "RadixSort": radix_sort
}

# -------------------------------------------------------------------------------------------------------------------------------- #
# Funciones de clave (key) para el ordenamiento

def sort_key_product(product):
    """
    Clave para ordenar productos:
    1) Año
    2) Título (alfabético si hay empate)
    """
    return (product["year"], product["title"].lower())


def sort_key_author(author_tuple):
    """
    Clave para ordenar autores (author_tuple = (autor, apariciones)):
    1) Número de apariciones
    2) Nombre (alfabético si hay empate)
    """
    return (author_tuple[1], author_tuple[0].lower())

# -------------------------------------------------------------------------------------------------------------------------------- #
# Requerimiento 1

def sort_products():
    archivo_unificado = load_products("data/processed/merged.bib")

    resultados = {}

    for nombre, algoritmo in algorithms.items():
        arr = archivo_unificado.copy()  # solo productos (dicts)

        # Inicia el conteo del tiempo del algortimo
        start = time.perf_counter()
        # Pasamos la función de clave directamente
        ordenados = algoritmo(arr, key=sort_key_product)
        # Finaliza el conteo del tiempo del algortimo
        end = time.perf_counter()

        resultados[nombre] = {
            "tiempo": end - start,
            "orden": ordenados
        }

        # Guardar resultado en CSV
        file_path = os.path.join(OUTPUT_DIR_PRODUCTS, f"{nombre.replace(' ', '_')}.csv")
        with open(file_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Year", "Title", "Type"])  # encabezado
            for prod in resultados[nombre]["orden"]:
                writer.writerow([prod["id"], prod["year"], prod["title"], prod["type"]])

        print(f"\n--- {nombre} ---")
        print(f"Tiempo: {resultados[nombre]['tiempo']:.6f} segundos")
        print(f"Resultados guardados en {file_path}")

    return resultados


# -------------------------------------------------------------------------------------------------------------------------------- #
# Requerimiento 2

"""
Función para dar con el cumplimiento del requerimimento:
Representar en un diagrama de barras y de manera ascendente los tiempos de los (12) algoritmos
de ordenamiento.
"""
def graph_times(resultados):
    """
    Genera un diagrama de barras ascendente con los tiempos de los algoritmos de ordenamiento.
    """
    # Extraer algoritmos y tiempos
    algoritmos = list(resultados.keys())
    tiempos = [resultados[n]["tiempo"] for n in algoritmos]

    # Ordenar de menor a mayor tiempo
    pares_ordenados = sorted(zip(algoritmos, tiempos), key=lambda x: x[1])
    algoritmos_ordenados, tiempos_ordenados = zip(*pares_ordenados)

    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.barh(algoritmos_ordenados, tiempos_ordenados)
    plt.xlabel("Tiempo de ejecución (segundos)")
    plt.ylabel("Algoritmo")
    plt.title("Comparación de tiempos de algoritmos de ordenamiento")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Guardar gráfico en archivo
    plt.tight_layout()
    plt.savefig("follow-ups/follow-up1/sorting_results/images/tiempos_algoritmos.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------------------------------------------------------------------- #
# Requerimiento 3

def sort_authors():
    # Leer archivo merged.bib
    with open("data/processed/merged.bib", "r", encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)

    # Extraer todos los autores de cada producto académico
    autores = []
    for entry in bib_database.entries:
        if "author" in entry:
            lista_autores = [a.strip() for a in entry["author"].split(" and ")]
            autores.extend(lista_autores)

    # Contar cuántas veces aparece cada autor
    conteo = Counter(autores)

    # Convertir a lista de tuplas (autor, apariciones)
    lista_autores = list(conteo.items())

    resultados = {}

    for nombre, algoritmo in algorithms.items():
        arr = lista_autores.copy()

        # Inicia el conteo del tiempo del algortimo
        start = time.perf_counter()
        # Pasamos la función de clave directamente
        ordenados = algoritmo(arr, key=sort_key_author)
        # Finaliza el conteo del tiempo del algortimo
        end = time.perf_counter()

        resultados[nombre] = {
            "tiempo": end - start,
            "orden": ordenados
        }

        # Guardar en CSV
        file_path = os.path.join(OUTPUT_DIR_AUTHORS, f"Autores_{nombre.replace(' ', '_')}.csv")
        with open(file_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Autor", "Apariciones"])
            for autor, apariciones in resultados[nombre]["orden"]:
                writer.writerow([autor, apariciones])

        print(f"\n--- {nombre} (Autores) ---")
        print(f"Tiempo: {resultados[nombre]['tiempo']:.6f} segundos")
        print(f"Resultados guardados en {file_path}")

    return resultados

# -------------------------------------------------------------------------------------------------------------------------------- #
"""
Función para dar con el cumplimiento del requerimimento:
Ordenar de manera ascendente los quince primeros autores con más apariciones en los
productos académicos.
"""
def graph_top15Authors():
    # Leer archivo merged.bib
    with open("data/processed/merged.bib", "r", encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)

    # Extraer todos los autores de cada producto académico
    autores = []
    for entry in bib_database.entries:
        if "author" in entry:
            # Los autores en BibTeX se separan con "and"
            lista_autores = [a.strip() for a in entry["author"].split(" and ")]
            autores.extend(lista_autores)

    # Contar cuántas veces aparece cada autor
    conteo = Counter(autores)

    # Ordenar por frecuencia ascendente y luego por nombre (para empates)
    ordenados = sorted(conteo.items(), key=sort_key_author)

    # Tomar solo los ultimos 15 (ya que está ordenado de forma ascendente)
    top15 = ordenados[-15:]

    # Se gráfica la información obtenida
    # Preparar datos para gráfica
    nombres = [autor for autor, _ in top15]
    valores = [apariciones for _, apariciones in top15]

    # Gráfica de barras
    plt.figure(figsize=(10, 6))
    plt.barh(nombres, valores, color="skyblue")
    plt.xlabel("Número de apariciones")
    plt.ylabel("Autores")
    plt.title("Top 15 autores con más apariciones (ascendente)")
    # Guardar gráfico en archivo
    plt.tight_layout()
    plt.savefig("follow-ups/follow-up1/top15_authors/images/top15_autores.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------------------------------------------------------------------- #

# Dirección para guardan los resultados de cada método ordenamiento
OUTPUT_DIR_PRODUCTS = "follow-ups/follow-up1/sorting_results"
OUTPUT_DIR_AUTHORS =  "follow-ups/follow-up1/top15_authors"
os.makedirs(OUTPUT_DIR_PRODUCTS, exist_ok=True)
os.makedirs(OUTPUT_DIR_AUTHORS, exist_ok=True)

if __name__ == "__main__":
    # Requerimiento 1 --------------------------------------------------------------------------- #
    # Ordenar los productos académicos segun los criterios (año y titulo) 
    # Se devuelve los resultados de tiempos de ejecución para cada algoritmo
    resultados = sort_products()

    # Requerimiento 2 --------------------------------------------------------------------------- #
    # Gráficar los tiempos de ejecución de cada algoritmo (para el ordenamiento de productos)
    graph_times(resultados)

    # Requerimiento 3 --------------------------------------------------------------------------- #
    # Ordenar los autores segun los criterios (numero de apariciones y nombre) 
    # Se devuelve los resultados de tiempos de ejecución para cada algoritmo
    resultados = sort_authors()
    # Gráficar los 15 autores con más apariciones (ordenados)
    graph_top15Authors()