"""
Aplicación de los 3 algoritmos jerárquicos y cálculo de métricas:
- average linkage (cosine)
- complete linkage (cosine)
- ward (SVD -> euclidean -> ward)
Genera dendrogramas y guarda métricas.
"""

from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import math

# Importación de librerías para visualización y clustering
import matplotlib.pyplot as plt

# Importación de funciones de clustering jerárquico y métricas
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

# Logger para registrar información y advertencias durante la ejecución
logger = logging.getLogger("clustering.algos")


def compute_condensed_cosine_distance(X):
    """
    Calcula la matriz de distancias coseno condensada para clustering.
    X: matriz TF-IDF (sparse o densa, filas=docs)
    Retorna la matriz condensada (formato pdist) con distancia = 1 - similitud coseno.
    ADVERTENCIA: convierte a densa si X es sparse (puede consumir mucha memoria).
    """
    try:
        from scipy.sparse import issparse
        if issparse(X):
            logger.debug("Convirtiendo matriz sparse a dense para pdist (puede usar mucha memoria).")
            Xd = X.toarray()
        else:
            Xd = np.asarray(X)
    except Exception:
        Xd = np.asarray(X)
    return pdist(Xd, metric="cosine")


def _shorten_labels(labels: List[str], max_len: int = 50) -> List[str]:
    """
    Acorta cada etiqueta si excede max_len (añade '...').
    Útil para evitar que los rótulos largos distorsionen los gráficos.
    """
    out = []
    for lab in labels:
        lab = str(lab)
        if len(lab) > max_len:
            out.append(lab[: max_len - 3].strip() + "...")
        else:
            out.append(lab)
    return out


def _dynamic_figsize_for_leaves(n_leaves: int, orientation: str = "top") -> Tuple[float, float]:
    """
    Calcula el tamaño de la figura (figsize) para el dendrograma según el número de hojas.
    orientation 'top' -> ancho dominante; 'right' -> alto dominante.
    """
    base = 6
    if orientation == "top":
        width = min(40, max(8, n_leaves * 0.25))  # evita anchos extremos
        height = base
    else:
        width = base
        height = min(60, max(6, n_leaves * 0.25))
    return (width, height)


def _plot_truncated_dendrogram(Z, labels_keys, out_path: Path, p: int = 40, max_label_len: int = 60, orientation: str = "top", dpi: int = 200):
    """
    Dibuja y guarda un dendrograma truncado (solo las últimas hojas).
    - p: cantidad de hojas finales a mostrar.
    - max_label_len: acorta rótulos largos.
    - orientation: 'top' o 'right'.
    """
    labels_to_use = _shorten_labels(labels_keys, max_len=max_label_len)
    figsize = _dynamic_figsize_for_leaves(min(p, len(labels_to_use)), orientation=orientation)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    kwargs = dict(labels=labels_to_use, truncate_mode="level", p=p, show_contracted=True) # truncate_mode: lastp"
    if orientation == "right":
        kwargs.update(dict(orientation="right", leaf_rotation=0))
    else:
        kwargs.update(dict(leaf_rotation=90))
    dendrogram(Z, **kwargs)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_average_linkage(X, labels_keys: List[str], out_dir: Path, p: int = 40, max_label_len: int = 60, orientation: str = "top"):
    """
    Ejecuta el algoritmo de average linkage usando distancia coseno.
    Guarda el dendrograma y retorna el modelo y la correlación cophenética.
    - p: número de hojas a mostrar (dendrograma truncado).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    D = compute_condensed_cosine_distance(X)
    Z = linkage(D, method="average")
    coph_corr, _ = cophenet(Z, D)
    logger.info("Average linkage: cophenetic correlation %.4f", coph_corr)

    out_path = out_dir / "dendrogram_average.png"
    try:
        _plot_truncated_dendrogram(Z, labels_keys, out_path, p=p, max_label_len=max_label_len, orientation=orientation)
    except Exception as ex:
        logger.warning("No se pudo generar dendrograma truncado: %s. Intentando dendrograma sin etiquetas.", ex)
        # fallback: genera dendrograma sin etiquetas para evitar problemas de memoria
        fig = plt.figure(figsize=(12, 6))
        dendrogram(Z, no_labels=True)
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return Z, coph_corr


def run_complete_linkage(X, labels_keys: List[str], out_dir: Path, p: int = 40, max_label_len: int = 60, orientation: str = "top"):
    """
    Ejecuta el algoritmo de complete linkage usando distancia coseno.
    Guarda el dendrograma y retorna el modelo y la correlación cophenética.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    D = compute_condensed_cosine_distance(X)
    Z = linkage(D, method="complete")
    coph_corr, _ = cophenet(Z, D)
    logger.info("Complete linkage: cophenetic correlation %.4f", coph_corr)

    out_path = out_dir / "dendrogram_complete.png"
    try:
        _plot_truncated_dendrogram(Z, labels_keys, out_path, p=p, max_label_len=max_label_len, orientation=orientation)
    except Exception as ex:
        logger.warning("Complete: fallback plot sin etiquetas: %s", ex)
        fig = plt.figure(figsize=(12, 6))
        dendrogram(Z, no_labels=True)
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return Z, coph_corr


def run_ward(X, labels_keys: List[str], out_dir: Path, n_components: int = 50, p: int = 40, max_label_len: int = 60, orientation: str = "top"):
    """
    Ejecuta el algoritmo Ward (requiere distancias euclidianas).
    Aplica TruncatedSVD a la matriz TF-IDF para obtener una representación densa.
    Devuelve el modelo, la correlación cophenética y el objeto SVD.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # TruncatedSVD acepta matrices sparse
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    Z = linkage(X_reduced, method="ward")  # ward acepta observaciones
    D_euc = pdist(X_reduced, metric="euclidean")
    coph_corr, _ = cophenet(Z, D_euc)
    logger.info("Ward linkage: cophenetic correlation %.4f (SVD components=%d)", coph_corr, n_components)

    out_path = out_dir / "dendrogram_ward.png"
    try:
        _plot_truncated_dendrogram(Z, labels_keys, out_path, p=p, max_label_len=max_label_len, orientation=orientation)
    except Exception as ex:
        logger.warning("Ward: fallback plot sin etiquetas: %s", ex)
        fig = plt.figure(figsize=(12, 6))
        dendrogram(Z, no_labels=True)
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return Z, coph_corr, svd


def evaluate_by_silhouette(X, Z, method: str, k_range: List[int], metric_for_silhouette: str = "cosine", use_reduced: Optional[np.ndarray] = None):
    """
    Evalúa la calidad de los clusters usando el coeficiente de silhouette para distintos valores de k.
    - Corta el dendrograma en varios k y calcula el promedio de silhouette.
    - Prefiltra los k que devuelven menos de 2 etiquetas únicas (evita errores/warnings).
    - Si use_reduced no es None (np.ndarray), silhouette se calcula sobre esa representación con métrica euclidiana.
    - Devuelve un diccionario k -> silhouette (NaN para los k inválidos).
    """
    results: Dict[int, float] = {}
    skipped_k: List[int] = []
    n_samples = None
    try:
        # Intentamos inferir n_samples del input X
        if hasattr(X, "shape"):
            n_samples = int(X.shape[0])
        else:
            n_samples = len(X)
    except Exception:
        n_samples = None

    for k in k_range:
        # 1) Comprobación rápida: k debe ser >=2 y <= n_samples-1 (si sabemos n_samples)
        if n_samples is not None:
            if k < 2 or k > max(1, n_samples - 1):
                skipped_k.append(k)
                results[k] = float("nan")
                continue

        # 2) Probar fcluster para ver cuántas etiquetas únicas produce
        try:
            labels = fcluster(Z, t=k, criterion="maxclust")
        except Exception as ex:
            # Si fcluster falla por cualquier razón, lo marcamos como inválido
            logger.debug("fcluster raised for method=%s k=%d: %s", method, k, ex)
            skipped_k.append(k)
            results[k] = float("nan")
            continue

        n_unique = len(set(labels))
        # Si fcluster devolvió menos de 2 etiquetas únicas, lo saltamos
        if n_unique < 2:
            skipped_k.append(k)
            results[k] = float("nan")
            continue

        # 3) Calcular silhouette únicamente para k válidos
        try:
            if use_reduced is not None:
                # Si tenemos representación reducida (Ward), usamos euclidiana sobre esa representación
                score = silhouette_score(use_reduced, labels, metric="euclidean")
            else:
                # Convertimos a densa (con precaución); silhouette puede aceptar 'cosine'
                X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                score = silhouette_score(X_dense, labels, metric=metric_for_silhouette)
        except Exception as ex:
            logger.debug("Error computing silhouette for method=%s k=%d: %s", method, k, ex)
            score = float("nan")

        results[k] = float(score)

    # Resumen: informar los k que fueron saltados (único log por llamada)
    if skipped_k:
        logger.info("Silhouette skipped for method=%s for k values (invalid or produced <2 labels): %s", method, skipped_k)

    return results
