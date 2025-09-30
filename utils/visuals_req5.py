#!/usr/bin/env python3
"""
utils/visuals_req5_improved.py

Versión mejorada del Requerimiento 5: análisis visual de la producción científica.

Mejoras implementadas:
1) Mapa de calor con mejor paleta de colores y diseño
2) Nube de palabras con filtrado de stopwords y mejor estética
3) Timeline mejorado con múltiples visualizaciones y diseño moderno
4) PDF con mejor layout y diseño profesional

Dependencias:
    pandas, bibtexparser, plotly, kaleido, wordcloud, reportlab, pycountry, matplotlib, seaborn, nltk
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías para visualización y exportación
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

# Parseo de BibTeX y procesamiento de texto
import bibtexparser
import pycountry

# Para filtrar stopwords en la nube de palabras
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('punkt', quiet=True)
    STOPWORDS_AVAILABLE = True
except ImportError:
    STOPWORDS_AVAILABLE = False

# Configuración de estilo para matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
    
    # Aliases comunes mejorados
    aliases = {
        "USA": "United States", "U.S.A.": "United States", "US": "United States",
        "UK": "United Kingdom", "U.K.": "United Kingdom", "England": "United Kingdom",
        "South Korea": "Korea, Republic of", "North Korea": "Korea, Democratic People's Republic of",
        "Russia": "Russian Federation", "Viet Nam": "Vietnam",
        "Czech Republic": "Czechia", "Holland": "Netherlands",
        "Taiwan": "Taiwan, Province of China", "Hong Kong": "Hong Kong",
        "Macau": "Macao", "Palestine": "Palestine, State of"
    }
    
    if t in aliases:
        t = aliases[t]
    
    # Prueba por nombre oficial
    for country in pycountry.countries:
        names = {country.name}
        if hasattr(country, "official_name"):
            names.add(getattr(country, "official_name"))
        
        if t.lower() in {n.lower() for n in names}:
            return country.alpha_3
    
    # Búsqueda "contains" mejorada
    for country in pycountry.countries:
        candidates = {country.name}
        if hasattr(country, "official_name"):
            candidates.add(getattr(country, "official_name"))
        
        for cand in candidates:
            if cand and len(cand) > 3 and cand.lower() in t.lower():
                return country.alpha_3
    
    return None


def extract_first_author_country(entry: dict) -> str:
    """
    Heurística mejorada para obtener el país del primer autor.
    """
    # Campos a revisar en orden de prioridad
    priority_fields = ["affiliation", "affiliations", "author+affiliation", "author_affiliation"]
    
    for field in priority_fields:
        aff = entry.get(field) or ""
        if aff:
            # Separar por diferentes delimitadores
            first_aff = re.split(r"[;,]| and | AND ", aff)[0].strip()
            iso3 = normalize_country(first_aff)
            if iso3:
                return iso3
    
    # Campos secundarios
    for field in ["address", "location", "publisher"]:
        address = entry.get(field, "")
        iso3 = normalize_country(address)
        if iso3:
            return iso3
    
    return "UNK"


def clean_text_for_wordcloud(text: str) -> str:
    """Limpia y procesa texto para la nube de palabras."""
    if not text:
        return ""
    
    # Remover caracteres especiales y números
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    if STOPWORDS_AVAILABLE:
        # Stopwords en inglés y español
        stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
        # Agregar palabras técnicas comunes poco informativas
        stop_words.update(['using', 'based', 'approach', 'method', 'algorithm', 'system', 
                          'paper', 'study', 'analysis', 'research', 'data', 'results',
                          'conclusion', 'introduction', 'methodology', 'discussion'])
        
        words = word_tokenize(text)
        text = ' '.join([word for word in words if word not in stop_words and len(word) > 2])
    
    return text


def to_dataframe(entries: list[dict]) -> pd.DataFrame:
    """Convierte entradas bibtex a un DataFrame optimizado."""
    records = []
    for e in entries:
        # Año con mejor parsing
        year = e.get("year")
        try:
            year_match = re.search(r'(19|20)\d{2}', str(year))
            year = int(year_match.group()) if year_match else None
        except Exception:
            year = None

        # Venue con mejor clasificación
        venue = (e.get("journal") or e.get("booktitle") or 
                e.get("publisher") or e.get("series") or "Otros")
        
        # Limpiar venue
        venue = re.sub(r'\{|\}', '', venue).strip()
        
        # Tipo de publicación
        pub_type = e.get("ENTRYTYPE", "article").lower()
        
        records.append({
            "title": e.get("title", ""),
            "authors": e.get("author", ""),
            "abstract": e.get("abstract", ""),
            "keywords": e.get("keywords", ""),
            "year": year,
            "venue": venue,
            "pub_type": pub_type,
            "first_author_country": extract_first_author_country(e),
        })
    
    df = pd.DataFrame.from_records(records)
    # Filtrar años válidos
    df = df[(df['year'] >= 1990) & (df['year'] <= 2030)]
    return df


def build_geo_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    """Genera un choropleth mejorado con mejor diseño."""
    counts = df["first_author_country"].value_counts().reset_index()
    counts.columns = ["iso_alpha", "count"]
    counts = counts[counts["iso_alpha"] != "UNK"]
    
    if counts.empty:
        # Imagen de fallback mejorada
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Distribución geográfica\n(sin datos de país disponibles)', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Crear mapa con Plotly mejorado
    fig = px.choropleth(
        counts,
        locations="iso_alpha",
        color="count",
        color_continuous_scale="Plasma",
        projection="natural earth",
        title="<b>Distribución Geográfica de Publicaciones</b><br><sub>Por país del primer autor</sub>",
        labels={"count": "Núm. Publicaciones", "iso_alpha": "País"}
    )
    
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        width=1200,
        height=700,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )
    
    fig.write_image(str(out_png), width=1200, height=700, scale=2)


def build_wordcloud(df: pd.DataFrame, out_png: Path) -> None:
    """Genera una nube de palabras mejorada."""
    corpus = []
    for _, row in df.iterrows():
        if isinstance(row["abstract"], str):
            corpus.append(clean_text_for_wordcloud(row["abstract"]))
        if isinstance(row["keywords"], str):
            keywords_clean = row["keywords"].replace(";", " ").replace(",", " ")
            corpus.append(clean_text_for_wordcloud(keywords_clean))
    
    text = " ".join(corpus)
    if not text.strip():
        text = "No data available"

    # WordCloud con mejor configuración
    wc = WordCloud(
        width=1600, 
        height=900, 
        background_color="white",
        colormap="viridis",
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10,
        prefer_horizontal=0.7,
        max_font_size=100
    )
    
    # Generar y guardar
    wordcloud_img = wc.generate(text)
    
    # Crear figura con matplotlib para mejor control
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wordcloud_img, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Nube de Palabras - Términos Más Frecuentes', 
                fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def build_timeline(df: pd.DataFrame, out_png: Path) -> None:
    """Genera timeline mejorado con múltiples visualizaciones."""
    # Preparar datos
    df_valid = df.dropna(subset=['year'])
    
    # Timeline general
    yearly_counts = df_valid.groupby('year').size().reset_index(name='count')
    
    # Top venues para el área apilada
    top_venues = df_valid['venue'].value_counts().head(5).index.tolist()
    df_top_venues = df_valid[df_valid['venue'].isin(top_venues)]
    venue_yearly = df_top_venues.groupby(['year', 'venue']).size().reset_index(name='count')
    
    # Crear subplots con Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Evolución Temporal Total",
            "Distribución por Tipo de Publicación", 
            "Top 5 Venues - Evolución",
            "Tendencia con Regresión"
        ),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"colspan": 1}, {"secondary_y": False}]]
    )
    
    # 1. Línea temporal principal con área
    fig.add_trace(
        go.Scatter(
            x=yearly_counts['year'],
            y=yearly_counts['count'],
            mode='lines+markers',
            name='Total Publicaciones',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.2)'
        ),
        row=1, col=1
    )
    
    # 2. Gráfico de pie por tipo de publicación
    pub_type_counts = df_valid['pub_type'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=pub_type_counts.index,
            values=pub_type_counts.values,
            name="Tipo Publicación",
            marker_colors=px.colors.qualitative.Set3
        ),
        row=1, col=2
    )
    
    # 3. Área apilada para top venues
    colors = px.colors.qualitative.Plotly
    for i, venue in enumerate(top_venues):
        venue_data = venue_yearly[venue_yearly['venue'] == venue]
        fig.add_trace(
            go.Scatter(
                x=venue_data['year'],
                y=venue_data['count'],
                mode='lines',
                name=venue[:30] + "..." if len(venue) > 30 else venue,
                stackgroup='one',
                line=dict(color=colors[i % len(colors)])
            ),
            row=2, col=1
        )
    
    # 4. Tendencia con regresión
    if len(yearly_counts) > 1:
        z = np.polyfit(yearly_counts['year'], yearly_counts['count'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=yearly_counts['year'],
                y=yearly_counts['count'],
                mode='markers',
                name='Datos Reales',
                marker=dict(color='#A23B72', size=10)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=yearly_counts['year'],
                y=p(yearly_counts['year']),
                mode='lines',
                name='Tendencia Linear',
                line=dict(color='#F18F01', width=3, dash='dash')
            ),
            row=2, col=2
        )
    
    # Actualizar layout
    fig.update_layout(
        height=800,
        width=1400,
        title_text="<b>Análisis Temporal de la Producción Científica</b>",
        title_x=0.5,
        title_font_size=18,
        font=dict(family="Arial, sans-serif", size=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Año", row=1, col=1)
    fig.update_yaxes(title_text="Número de Publicaciones", row=1, col=1)
    fig.update_xaxes(title_text="Año", row=2, col=1)
    fig.update_yaxes(title_text="Publicaciones (Acumulado)", row=2, col=1)
    fig.update_xaxes(title_text="Año", row=2, col=2)
    fig.update_yaxes(title_text="Número de Publicaciones", row=2, col=2)
    
    fig.write_image(str(out_png), width=1400, height=800, scale=2)


def create_summary_stats(df: pd.DataFrame) -> dict:
    """Crea estadísticas resumidas para el reporte."""
    stats = {
        'total_publications': len(df),
        'year_range': f"{df['year'].min():.0f} - {df['year'].max():.0f}",
        'countries_count': df[df['first_author_country'] != 'UNK']['first_author_country'].nunique(),
        'venues_count': df['venue'].nunique(),
        'top_country': df[df['first_author_country'] != 'UNK']['first_author_country'].mode().iloc[0] if len(df[df['first_author_country'] != 'UNK']) > 0 else 'N/A',
        'top_venue': df['venue'].mode().iloc[0],
        'publications_per_year': df.groupby('year').size().mean(),
        'peak_year': df['year'].mode().iloc[0] if not df.empty else 'N/A'
    }
    return stats


def export_pdf_improved(images: list[Path], out_pdf: Path, stats: dict) -> None:
    """Exporta un PDF mejorado con estadísticas y mejor diseño."""
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4,
                          rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#A23B72'),
        spaceAfter=15
    )
    
    story = []
    
    # Título del reporte
    story.append(Paragraph("Análisis Visual de la Producción Científica", title_style))
    story.append(Spacer(1, 20))
    
    # Estadísticas resumidas
    story.append(Paragraph("Estadísticas Generales", subtitle_style))
    
    stats_text = f"""
    <para>
    <b>Total de Publicaciones:</b> {stats['total_publications']}<br/>
    <b>Rango Temporal:</b> {stats['year_range']}<br/>
    <b>Países Representados:</b> {stats['countries_count']}<br/>
    <b>Venues Únicos:</b> {stats['venues_count']}<br/>
    <b>País Más Productivo:</b> {stats['top_country']}<br/>
    <b>Venue Principal:</b> {stats['top_venue'][:50]}{'...' if len(stats['top_venue']) > 50 else ''}<br/>
    <b>Promedio Anual:</b> {stats['publications_per_year']:.1f} publicaciones<br/>
    <b>Año Pico:</b> {stats['peak_year']}
    </para>
    """
    
    story.append(Paragraph(stats_text, styles['Normal']))
    story.append(PageBreak())
    
    # Agregar imágenes
    titles = [
        "Distribución Geográfica por País",
        "Nube de Palabras - Términos Frecuentes", 
        "Análisis Temporal de Publicaciones"
    ]
    
    for i, (img_path, title) in enumerate(zip(images, titles)):
        if img_path.exists():
            story.append(Paragraph(title, subtitle_style))
            story.append(Spacer(1, 10))
            
            # Ajustar imagen al tamaño de página
            img = Image(str(img_path))
            img.drawHeight = 400
            img.drawWidth = 500
            story.append(img)
            story.append(Spacer(1, 20))
            
            if i < len(images) - 1:  # No añadir page break después de la última imagen
                story.append(PageBreak())
    
    doc.build(story)


def run(bib_path: Path, out_dir: Path) -> dict:
    """Función principal mejorada."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando datos bibliográficos...")
    entries = load_bib(bib_path)
    df = to_dataframe(entries)
    
    print(f"Procesando {len(df)} entradas...")
    
    # Rutas de salida
    heatmap_png = out_dir / "req5_geo_heatmap.png"
    wordcloud_png = out_dir / "req5_wordcloud.png" 
    timeline_png = out_dir / "req5_timeline.png"
    pdf_path = out_dir / "req5_report.pdf"

    print("Generando mapa de calor geográfico...")
    build_geo_heatmap(df, heatmap_png)
    
    print("Generando nube de palabras...")
    build_wordcloud(df, wordcloud_png)
    
    print("Generando análisis temporal...")
    build_timeline(df, timeline_png)
    
    print("Calculando estadísticas...")
    stats = create_summary_stats(df)
    
    print("Exportando PDF...")
    export_pdf_improved([heatmap_png, wordcloud_png, timeline_png], pdf_path, stats)

    # CSV de debug mejorado
    debug_csv = out_dir / "req5_dataframe_debug.csv"
    df.to_csv(debug_csv, index=False)
    
    print("¡Proceso completado!")
    
    return {
        "heatmap_png": str(heatmap_png),
        "wordcloud_png": str(wordcloud_png), 
        "timeline_png": str(timeline_png),
        "pdf": str(pdf_path),
        "debug_csv": str(debug_csv),
        "stats": stats
    }


def main_visuals_req5():
    bib_path = Path("data/processed/merged.bib")
    out_dir = Path("outputs/biblometric_info")
    
    if not bib_path.exists():
        print(f"Error: No se encuentra el archivo {bib_path}")
        return
    
    results = run(bib_path, out_dir)
    print("\n=== RESULTADOS ===")
    for key, value in results.items():
        if key != 'stats':
            print(f"{key}: {value}")
    
    print("\n=== ESTADÍSTICAS ===")
    for key, value in results['stats'].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main_visuals_req5()