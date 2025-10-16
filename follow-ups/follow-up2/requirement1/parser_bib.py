import bibtexparser

def parse_bib_file(file_path="data/processed/merged.bib"):
    """
    Lee y parsea el archivo merged.bib, devolviendo una lista de diccionarios
    con los campos relevantes de cada artículo (id, título, autores, año, resumen, keywords).
    """
    with open(file_path, encoding="utf-8") as f:
        db = bibtexparser.load(f)

    docs = []
    for entry in db.entries:
        doc = {
            "id": entry.get("ID", "").strip(),
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "authors": [a.strip() for a in entry.get("author", "").replace("\n", " ").split(" and ") if a.strip()],
            "year": int(entry["year"]) if "year" in entry and entry["year"].isdigit() else None,
            "abstract": entry.get("abstract", "").replace("\n", " ").strip(),
            "keywords": [k.strip() for k in entry.get("keywords", "").replace("\n", " ").split(",") if k.strip()]
        }
        docs.append(doc)

    return docs
