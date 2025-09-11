"""
Script principal: orquesta lectura, preprocesamiento, ejecución de los 3 algoritmos,
evaluación (cophenet + silhouette) y guardado de resultados.

Salidas en data/processed/:
 - dendrogram_average.png
 - dendrogram_complete.png
 - dendrogram_ward.png
 - clustering_metrics.csv

Ejecución:
    python3 utils/clustering/run_all.py

"""

from pathlib import Path  # Para manejo de rutas de archivos
import logging  # Para registro de mensajes
import csv  # Para guardar resultados en CSV
import math  # Para operaciones matemáticas
import numpy as np  # Para manejo de matrices
from typing import Optional  # Para anotaciones de tipo
import sys

# Se añade la carpeta raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importación de funciones de preprocesamiento y vectorización
from utils.clustering.preprocess_and_vectorize import (
    load_abstracts_from_bib,
    preprocess_abstracts,
    vectorize_texts,
)
# Importación de algoritmos de clustering y métricas
from utils.clustering.hierarchical_algos import (
    run_average_linkage,
    run_complete_linkage,
    run_ward,
    evaluate_by_silhouette,
)

# Inicialización del logger
logger = logging.getLogger("clustering.run")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def choose_best_k_from_sil(sil_dict: dict) -> Optional[int]:
    """
    Escoge el valor de k que tiene el mayor score de silhouette, ignorando valores NaN.
    Devuelve None si no hay valores válidos.
    """
    valid = {k: v for k, v in sil_dict.items() if v is not None and (not (isinstance(v, float) and math.isnan(v)))}
    if not valid:
        return None
    best_k = max(valid.keys(), key=lambda kk: valid[kk])
    return best_k


def safe_toarray(X):
    """
    Convierte X a matriz densa (numpy array) con precaución.
    Si la matriz es muy grande, muestra advertencia por posible uso intensivo de memoria.
    """
    try:
        from scipy.sparse import issparse
        if issparse(X):
            n, m = X.shape
            if n * m > 20_000_000:  # heurístico: 20M celdas ~ posible uso intensivo de memoria
                logger.warning("Conversion to dense may be costosa: n=%d, m=%d. Procediendo de todas formas.", n, m)
            return X.toarray()
    except Exception:
        pass
    return np.asarray(X)


def main():
    # Obtiene la ruta raíz del proyecto
    pr = Path(__file__).resolve().parents[2]  # project root
    # Define la ruta al archivo merged.bib y al directorio de salida
    merged_bib = pr / "data" / "processed" / "merged.bib"
    out_dir = pr / "outputs" / "clustering_&_dendograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 Carga y preprocesa los abstracts
    raw_pairs = load_abstracts_from_bib(merged_bib)  # Lee abstracts del archivo bibtex
    if not raw_pairs:
        logger.error("No se encontraron abstracts en %s. Abortando.", merged_bib)
        return
    clean_pairs = preprocess_abstracts(raw_pairs)  # Limpia y lematiza los abstracts
    keys, vectorizer, X = vectorize_texts(clean_pairs)  # Vectoriza los textos con TF-IDF

    # 2 Ejecuta los algoritmos de clustering jerárquico y guarda los dendrogramas
    Z_avg, coph_avg = run_average_linkage(X, keys, out_dir, p=5, max_label_len=60, orientation="top")  # Average linkage
    Z_comp, coph_comp = run_complete_linkage(X, keys, out_dir, p=5, max_label_len=60, orientation="top")  # Complete linkage
    Z_ward, coph_ward, svd = run_ward(X, keys, out_dir, n_components=2, p=5, max_label_len=60, orientation="top")  # Ward linkage

    # 3 Evalúa la calidad de los clusters usando silhouette para k en 2..8 (robusto contra NaNs)
    k_range = list(range(2, 9))  # Rango de valores de k a probar
    # Calcula silhouette para average linkage
    sil_avg = evaluate_by_silhouette(X, Z_avg, "average", k_range, metric_for_silhouette="cosine")
    # Calcula silhouette para complete linkage
    sil_comp = evaluate_by_silhouette(X, Z_comp, "complete", k_range, metric_for_silhouette="cosine")

    # Para Ward: intenta obtener la representación reducida y evalúa silhouette sobre ella
    try:
        X_dense = safe_toarray(X)  # Convierte a matriz densa si es necesario
        X_reduced = svd.transform(X_dense)  # Aplica SVD para reducción de dimensionalidad
    except Exception:
        try:
            X_reduced = svd.transform(X)  # Intenta transformar directamente si falla la conversión
        except Exception as ex:
            logger.warning("No se pudo obtener X_reduced para Ward: %s", ex)
            X_reduced = None

    # Calcula silhouette para Ward usando la representación reducida
    sil_ward = evaluate_by_silhouette(X, Z_ward, "ward", k_range, metric_for_silhouette="euclidean", use_reduced=X_reduced)

    # 4 Guarda las métricas en un archivo CSV
    metrics_path = out_dir / "clustering_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "cophenetic_correlation", "k", "silhouette"])
        for k in k_range:
            writer.writerow(["average", coph_avg, k, sil_avg.get(k, "")])
        for k in k_range:
            writer.writerow(["complete", coph_comp, k, sil_comp.get(k, "")])
        for k in k_range:
            writer.writerow(["ward", coph_ward, k, sil_ward.get(k, "")])
    logger.info("Saved metrics CSV: %s", metrics_path)

    # 5 Muestra en el log el mejor k por método (no se exporta)
    def log_best(name, sil_dict):
        best_k = choose_best_k_from_sil(sil_dict)
        if best_k is None:
            logger.info("No valid silhouette found for %s.", name)
        else:
            logger.info("Best silhouette (%s) k=%s score=%.4f", name, best_k, sil_dict.get(best_k, float("nan")))

    # Muestra las correlaciones cophenéticas y el mejor k para cada método
    logger.info("Cophenetic correlations -> average: %.4f, complete: %.4f, ward: %.4f", coph_avg, coph_comp, coph_ward)
    log_best("average", sil_avg)
    log_best("complete", sil_comp)
    log_best("ward", sil_ward)

    logger.info("Resultados guardados en: %s", out_dir)


# Punto de entrada principal del script
if __name__ == "__main__":
    main()
