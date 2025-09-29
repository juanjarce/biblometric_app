import os
import streamlit as st
from scraper.acm_scraper3 import scrape_acm_bibtex
from scraper.ieee_scraper import scrape_ieee_bibtex
from utils.keywords_analizer import main_keywords_analizer
from utils.clustering.run_all import main_dendrograms_analizer
from utils.visuals_req5 import main_visuals_req5
import subprocess, time

st.title("Menú Proyecto")

# --- Scraper ACM ---
st.subheader("Scraper ACM")
start_acm = st.number_input("Página inicio (ACM)", min_value=0, step=1, key="acm_start")
count_acm = st.number_input("Cantidad páginas (ACM)", min_value=1, step=1, key="acm_count")
if st.button("Ejecutar Scraper ACM"):
    for i in range(start_acm, start_acm + count_acm):
        scrape_acm_bibtex(i)
        time.sleep(3)
    st.success("Scraper ACM completado ✅")

# --- Scraper IEEE ---
st.subheader("Scraper IEEE")
start_ieee = st.number_input("Página inicio (IEEE)", min_value=1, step=1, key="ieee_start")
count_ieee = st.number_input("Cantidad páginas (IEEE)", min_value=1, step=1, key="ieee_count")
if st.button("Ejecutar Scraper IEEE"):
    for i in range(start_ieee, start_ieee + count_ieee):
        scrape_ieee_bibtex(i)
        time.sleep(3)
    st.success("Scraper IEEE completado ✅")

# --- Merge Bib ---
st.subheader("Merge Bib")
if st.button("Ejecutar Merge Bib"):
    subprocess.run(["python3", "utils/merge_bib.py", "--ieee-dir", "data/raw/IEEE",
                    "--acm-dirs", "data/raw/ACM3", "--out-dir", "data/processed"])
    st.success("Merge completado ✅")

# --- Comparar Artículos ---
st.subheader("Comparar artículos")
ids = st.text_input("Ingrese los IDs (ej: merged1,merged3)")

if st.button("Ejecutar Comparación"):
    if ids.strip():
        result = subprocess.run(
            ["python3", "utils/compare_articles.py", "--compare_ids", ids],
            capture_output=True,
            text=True
        )

        # Mostrar salida estándar
        if result.stdout:
            st.text_area("Resultados de la comparación:", result.stdout, height=300)

        # Mostrar errores si existen
        if result.stderr:
            st.error(f"Errores detectados:\n{result.stderr}")

        st.success(f"Comparación completada ✅ con IDs: {ids}")
    else:
        st.warning("Por favor ingrese al menos un ID.")

# --- Analizador Keywords ---
st.subheader("Analizador Keywords")
if st.button("Ejecutar Analizador Keywords"):
    # Ejecutar análisis
    main_keywords_analizer()
    st.success("Análisis de keywords completado ✅")

    # Ruta donde se guardan las imágenes
    output_dir = "outputs/keywords_analizer"

    # Buscar archivos PNG en la carpeta
    images = [f for f in os.listdir(output_dir) if f.endswith(".png")]

    if images:
        st.write("### Resultados del Análisis")
        for img in images:
            st.image(os.path.join(output_dir, img), caption=img)
    else:
        st.warning("No se encontraron imágenes en la carpeta de resultados.")

st.subheader("Analizador de Similitud con Dendrogramas")
if st.button("Ejecutar Dendrogramas"):
    # Ejecutar análisis
    main_dendrograms_analizer()
    st.success("Dendrogramas completados ✅")

    # Ruta donde se guardan resultados
    output_dir = "outputs/clustering_&_dendograms"

    # Buscar imágenes y csv
    images = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    # Mostrar imágenes
    if images:
        st.write("### Resultados en Imágenes")
        for img in images:
            st.image(os.path.join(output_dir, img), caption=img)

    # Mostrar CSV para descarga
    if csv_files:
        st.write("### Resultados en CSV")
        for csv in csv_files:
            file_path = os.path.join(output_dir, csv)

            # Mostrar tabla
            import pandas as pd
            df = pd.read_csv(file_path)
            st.dataframe(df)

# --- Visualización Req 5 ---
st.subheader("Visualización Información Bibliométrica (Req 5)")
if st.button("Ejecutar Visualización Req 5"):
    # Ejecutar análisis
    main_visuals_req5()
    st.success("Visualización generada ✅")

    # Ruta de salida
    output_dir = "outputs/biblometric_info"

    # Buscar archivos generados
    images = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith(".pdf")]
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    # Mostrar imágenes
    if images:
        st.write("### Resultados en Imágenes")
        for img in images:
            st.image(os.path.join(output_dir, img), caption=img)

    # Mostrar CSV (tabla y descarga)
    if csv_files:
        st.write("### Resultados en CSV")
        for csv in csv_files:
            file_path = os.path.join(output_dir, csv)

            # Mostrar tabla
            import pandas as pd
            df = pd.read_csv(file_path)
            st.dataframe(df)

    # Mostrar PDF para descarga
    if pdf_files:
        st.write("### Reportes en PDF")
        for pdf in pdf_files:
            file_path = os.path.join(output_dir, pdf)
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"Descargar {pdf}",
                    data=f,
                    file_name=pdf,
                    mime="application/pdf"
                )