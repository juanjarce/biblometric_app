import bibtexparser

def load_abstracts_from_bib(file_path="data/processed/merged.bib"):
    """
    Carga los abstracts desde un archivo .bib y los devuelve
    en un diccionario con la estructura:
        {
            "merged1": "Texto del abstract...",
            "merged2": "Texto del abstract...",
            ...
        }
    Solo incluye artículos que tengan abstract no vacío.
    """
    with open(file_path, encoding="utf-8") as f:
        db = bibtexparser.load(f)

    abstracts = {}
    for entry in db.entries:
        article_id = entry.get("ID", "").strip()
        abstract = entry.get("abstract", "").replace("\n", " ").strip()
        if article_id and abstract:
            abstracts[article_id] = abstract

    print(f"Se cargaron {len(abstracts)} abstracts con contenido desde '{file_path}'.")
    return abstracts
