"""
Preprocesamiento y vectorización de abstracts.
- Lee merged.bib y extrae abstracts.
- Limpia, tokeniza, lematiza (NLTK), remueve stopwords.
- Vectoriza con TfidfVectorizer.
"""

from pathlib import Path
import re
import logging
from typing import List, Tuple, Optional

# Librerías para manejo de archivos bibtex, procesamiento de texto y vectorización
import bibtexparser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Logger para registrar información y advertencias
logger = logging.getLogger("clustering.preprocess")

# Intentamos descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Descargando recursos NLTK (wordnet, stopwords)...")
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("omw-1.4")

# Inicialización de lematizador y lista de stopwords
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))  # por defecto en inglés; ajustar si necesario

def load_abstracts_from_bib(bib_path: Path) -> List[Tuple[str, str]]:
    """
    Lee el archivo merged.bib y devuelve una lista de tuplas (clave, abstract).
    Omite entradas que no tienen abstract.
    """
    text = bib_path.read_text(encoding="utf-8", errors="ignore")
    db = bibtexparser.loads(text)
    results = []
    for entry in db.entries:
        abstract = entry.get("abstract", "")
        key = entry.get("ID", entry.get("key", ""))
        if abstract and abstract.strip():
            results.append((key, abstract.strip()))
    logger.info("Cargados %d abstracts desde %s", len(results), bib_path)
    return results

def simple_clean(text: str) -> str:
    """
    Realiza limpieza básica del texto:
    - Elimina URLs
    - Elimina caracteres no alfanuméricos
    - Convierte a minúsculas
    - Elimina espacios extra
    """
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_and_remove_stopwords(text: str) -> str:
    """
    Tokeniza el texto, lematiza cada palabra y elimina stopwords.
    Devuelve el texto procesado.
    """
    tokens = text.split()
    lemmas = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        try:
            lem = lemmatizer.lemmatize(t)
        except Exception:
            lem = t
        if lem and len(lem) > 1:
            lemmas.append(lem)
    return " ".join(lemmas)

def preprocess_abstracts(raw_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Aplica limpieza y lematización a cada abstract.
    Devuelve lista de tuplas (clave, texto limpio).
    """
    out = []
    for key, text in raw_pairs:
        c = simple_clean(text)
        c = lemmatize_and_remove_stopwords(c)
        if c:
            out.append((key, c))
    logger.info("Preprocesados %d abstracts", len(out))
    return out

def vectorize_texts(clean_pairs: List[Tuple[str, str]], max_features: int = 20000, ngram_range=(1,2), min_df=2):
    """
    Vectoriza los textos limpios usando TF-IDF.
    Devuelve las claves, el vectorizador y la matriz TF-IDF resultante.
    """
    keys = [k for k, _ in clean_pairs]
    texts = [t for _, t in clean_pairs]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(texts)  # matriz dispersa
    logger.info("TF-IDF shape: %s", X.shape)
    return keys, vectorizer, X
