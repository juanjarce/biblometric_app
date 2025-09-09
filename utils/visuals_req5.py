
#!/usr/bin/env python3
"""
utils/visuals_req5.py

Requerimiento 5 del proyecto: análisis visual de la producción científica.

Funciones:
1) Mapa de calor (choropleth) de la distribución geográfica por país del primer autor.
2) Nube de palabras con términos más frecuentes en abstracts y keywords (dinámica: se recalcula con los datos actuales).
3) Línea temporal de publicaciones por año (global) y por revista (múltiples líneas).
4) Exportación de las tres visualizaciones a un único PDF.

Uso:
    python -m utils.visuals_req5 --bib data/processed/merged.bib

Dependencias (agregadas a requirements.txt):
    pandas, bibtexparser, plotly, kaleido, wordcloud, reportlab, pycountry
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Librerías para visualización y exportación
import plotly.express as px
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Parseo de BibTeX
import bibtexparser
import pycountry


def load_bib(bib_path: str | Path) -> list[dict]:
    """Carga el archivo .bib y retorna una lista de entradas (dicts)."""
    bib_path = Path(bib_path)
    with open(bib_path, "r", encoding="utf-8", errors="ignore") as f:
        db = bibtexparser.load(f)
    return db.entries


def normalize_country(text: str) -> str | None:
    """Intenta mapear un fragmento de texto a un nombre de país estandarizado (ISO-3)."""
    if not text:
        return None
    t = text.strip()
    # Prueba por nombre oficial
    for country in pycountry.countries:
        names = {country.name}
        if hasattr(country, "official_name"):
            names.add(getattr(country, "official_name"))
        # Include common aliases
        aliases = {
            "USA": "United States",
            "U.S.A.": "United States",
            "UK": "United Kingdom",
            "U.K.": "United Kingdom",
            "South Korea": "Korea, Republic of",
            "North Korea": "Korea, Democratic People's Republic of",
            "Russia": "Russian Federation",
            "Viet Nam": "Vietnam",
            "Czech Republic": "Czechia",
        }
        if t in aliases:
            t = aliases[t]
        if t.lower() in {n.lower() for n in names}:
            return country.alpha_3
    # Búsqueda "contains"
    for country in pycountry.countries:
        candidates = {country.name}
        if hasattr(country, "official_name"):
            candidates.add(getattr(country, "official_name"))
        candidates |= {"United States" if country.alpha_2 == "US" else ""}
        for cand in list(candidates):
            if cand and cand.lower() in t.lower():
                return country.alpha_3
    return None


def extract_first_author_country(entry: dict) -> str:
    """
    Heurística para obtener el país del primer autor.
    Se intenta por los campos más comunes: 'affiliation', 'author', 'address'.
    Si no se encuentra país, retorna 'UNK'.
    """
    # 1) affiliation del primer autor (si viene separada por ';' o ' and ')
    for field in ("affiliation", "affiliations", "author+affiliation", "author_affiliation"):
        aff = entry.get(field) or ""
        if aff:
            first_aff = re.split(r";| and ", aff)[0]
            iso3 = normalize_country(first_aff)
            if iso3:
                return iso3

    # 2) address a veces tiene país del evento o editorial (mejor que nada)
    address = entry.get("address") or entry.get("location") or ""
    iso3 = normalize_country(address)
    if iso3:
        return iso3

    # 3) keywords/abstract (poco probable, último recurso)
    for field in ("keywords", "abstract"):
        txt = entry.get(field, "")
        iso3 = normalize_country(txt)
        if iso3:
            return iso3

    return "UNK"


def to_dataframe(entries: list[dict]) -> pd.DataFrame:
    """Convierte entradas bibtex a un DataFrame compacto para las gráficas."""
    records = []
    for e in entries:
        # Año
        year = e.get("year")
        try:
            year = int(re.findall(r"\d{4}", str(year))[0])
        except Exception:
            year = None

        # Revista o libro de actas
        venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or "Desconocido"

        # Autores (texto)
        authors = e.get("author", "")

        # Abstract y keywords
        abstract = e.get("abstract", "")
        keywords = e.get("keywords", "")

        # País del primer autor (ISO-3 o UNK)
        first_country = extract_first_author_country(e)

        records.append({
            "title": e.get("title", ""),
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords,
            "year": year,
            "venue": venue,
            "first_author_country": first_country,
        })
    return pd.DataFrame.from_records(records)


def build_geo_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    """Genera un choropleth por país (conteo de publicaciones por primer autor)."""
    counts = df["first_author_country"].value_counts().reset_index()
    counts.columns = ["iso_alpha", "count"]
    counts = counts[counts["iso_alpha"] != "UNK"]
    if counts.empty:
        # Genera una imagen de fallback
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Distribución geográfica (sin datos de país)")
        plt.xlabel("No se encontró país en los metadatos")
        plt.ylabel("Publicaciones")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return

    fig = px.choropleth(
        counts,
        locations="iso_alpha",
        color="count",
        color_continuous_scale="Viridis",
        projection="natural earth",
        title="Distribución geográfica por país (primer autor)",
    )
    fig.write_image(str(out_png))  # requiere 'kaleido' instalado


def build_wordcloud(df: pd.DataFrame, out_png: Path) -> None:
    """Genera una nube de palabras a partir de abstracts + keywords."""
    corpus = []
    for _, row in df.iterrows():
        if isinstance(row["abstract"], str):
            corpus.append(row["abstract"])
        if isinstance(row["keywords"], str):
            corpus.append(row["keywords"].replace(";", " ").replace(",", " "))
    text = " ".join(corpus)
    if not text.strip():
        text = "No data"

    wc = WordCloud(width=1600, height=900, background_color="white")
    img = wc.generate(text).to_image()
    img.save(out_png)


def build_timeline(df: pd.DataFrame, out_png: Path) -> None:
    """Genera líneas temporales: total por año y por revista (múltiples líneas)."""
    # Conteo por año
    by_year = df.groupby("year").size().reset_index(name="count").dropna()
    # Conteo por año y revista (top 7 revistas)
    top_venues = df["venue"].value_counts().head(7).index.tolist()
    df_top = df[df["venue"].isin(top_venues)]
    by_year_venue = df_top.groupby(["year", "venue"]).size().reset_index(name="count").dropna()

    # Usamos plotly para salvar imagen estática
    import plotly.graph_objects as go

    fig = go.Figure()
    # Línea total
    fig.add_trace(go.Scatter(x=by_year["year"], y=by_year["count"], mode="lines+markers", name="Total"))
    # Líneas por revista
    for v in top_venues:
        sub = by_year_venue[by_year_venue["venue"] == v]
        fig.add_trace(go.Scatter(x=sub["year"], y=sub["count"], mode="lines+markers", name=v))

    fig.update_layout(
        title="Línea temporal de publicaciones (Total y por revista)",
        xaxis_title="Año",
        yaxis_title="Publicaciones",
    )
    fig.write_image(str(out_png))  # requiere 'kaleido'


def export_pdf(images: list[Path], out_pdf: Path) -> None:
    """Exporta las imágenes al PDF en páginas separadas."""
    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter
    margin = 40
    for img_path in images:
        img = ImageReader(str(img_path))
        iw, ih = img.getSize()
        # Ajuste manteniendo aspecto
        scale = min((width - 2*margin) / iw, (height - 2*margin) / ih)
        new_w, new_h = iw * scale, ih * scale
        x = (width - new_w) / 2
        y = (height - new_h) / 2
        c.drawImage(img, x, y, new_w, new_h)
        c.showPage()
    c.save()


def run(bib_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_bib(bib_path)
    df = to_dataframe(entries)

    # Salidas
    heatmap_png = out_dir / "req5_geo_heatmap.png"
    wordcloud_png = out_dir / "req5_wordcloud.png"
    timeline_png = out_dir / "req5_timeline.png"
    pdf_path = out_dir / "req5_report.pdf"

    build_geo_heatmap(df, heatmap_png)
    build_wordcloud(df, wordcloud_png)
    build_timeline(df, timeline_png)
    export_pdf([heatmap_png, wordcloud_png, timeline_png], pdf_path)

    # CSV auxiliar por si se requiere depurar
    df.to_csv(out_dir / "req5_dataframe_debug.csv", index=False)

    return {
        "heatmap_png": str(heatmap_png),
        "wordcloud_png": str(wordcloud_png),
        "timeline_png": str(timeline_png),
        "pdf": str(pdf_path),
        "debug_csv": str(out_dir / "req5_dataframe_debug.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description="Requerimiento 5: visualizaciones y PDF")
    parser.add_argument("--bib", default="data/processed/merged.bib", help="Ruta al .bib unificado")
    parser.add_argument("--out", default="outputs", help="Directorio de salida")
    args = parser.parse_args()

    bib_path = Path(args.bib)
    out_dir = Path(args.out)
    results = run(bib_path, out_dir)
    print("Generado:", results)


if __name__ == "__main__":
    main()
