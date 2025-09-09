## Cómo activar el entorno virtual
python3 -m venv venv

## Para la ejecución de la API

Antes de ejecutar el scraper o cualquier script del proyecto, asegúrate de activar el entorno virtual:

### En macOS / Linux
```bash
source venv/bin/activate

### En Windows (CMD)
venv\Scripts\activate.bat

### En Windows (PowerShell)
venv\Scripts\Activate.ps1

### Para desactivar el entorno virtual
deactivate

## Requerimiento 5 — Visualizaciones y PDF

Este módulo genera:
1. **Mapa de calor geográfico** por país del primer autor (choropleth).
2. **Nube de palabras** con términos frecuentes (abstracts + keywords).
3. **Línea temporal** de publicaciones por año (total) y por revista (múltiples líneas).
4. **Exportación a PDF** con las tres figuras.

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

> Nota: Para guardar imágenes estáticas de Plotly se usa **kaleido**.

### Uso
```bash
# Desde la raíz del proyecto
python -m utils.visuals_req5 --bib data/processed/merged.bib --out outputs
```

Los archivos generados se almacenan en `outputs/`:
- `req5_geo_heatmap.png`
- `req5_wordcloud.png`
- `req5_timeline.png`
- `req5_report.pdf`
- `req5_dataframe_debug.csv` (opcional para inspeccionar los datos que alimentan las gráficas)

### Suposiciones de datos
- Se intenta detectar el país del primer autor a partir de campos como `affiliation`, `address`, `location`.
- Si no es posible inferir el país, el registro se marca como `UNK` y el mapa lo omitirá.
- La nube de palabras se recalcula cada vez que se ejecute el comando, por lo que **es dinámica** al agregar nuevos estudios al `.bib`.
