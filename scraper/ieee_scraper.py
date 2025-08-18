from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from datetime import datetime
from pathlib import Path
import shutil
import time
import os

def scrape_ieee_bibtex(page: int):
    """
    Lanzar una ventana en Firefox browser, navegar a la Libreria Digital de IEEE,
    aceptar las cookies, seleccionar la opcion de descargar todos los resultads, obtenerlas citaciones en formato BibTeX 
    """
    download_dir = str(Path("downloads").resolve())
    output_dir = Path("data/raw/IEEE")
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuración de Firefox
    firefox_options = Options()
    configure_firefox(firefox_options, download_dir)

    # Iniciar navegador
    driver = webdriver.Firefox(options=firefox_options)
    wait = WebDriverWait(driver, 10)

    # Ir a la página de IEEE
    url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=computational%20thinking&highlight=true&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber={page}&returnFacets=ALL"
    driver.get(url)

    try:
        # Si es la primera pagina, se acepta las cookies
        print("Esperando para aceptar las cookies...")
        cookie_button = wait.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR, "button.osano-cm-accept-all.osano-cm-buttons__button.osano-cm-button.osano-cm-button--type_accept"
            ))
        )
        cookie_button.click()
        print("Cookies aceptadas")
        time.sleep(2)

        # Se seleccionan todos los resultados de la pagina por la busquedá
        print("Seleccionando todos los resultados...")
        select_all_checkbox = wait.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "input.xpl-checkbox-default.results-actions-selectall-checkbox"
            ))
        )
        select_all_checkbox.click()
        time.sleep(2)

        # Se abre el modal de exportación
        print("Abriendo el modal de exportación...")
        export_button = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[@class='xpl-btn-primary' and normalize-space(text())='Export']"
            ))
        )
        export_button.click()
        time.sleep(2)

        print("Accediendo al modal de 'Citaciones'...")
        citations_button = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//a[@class='nav-link' and normalize-space(text())='Citations']"
            ))
        )
        citations_button.click()
        time.sleep(2)

        print("Seleccionando formato de descarga...")
        # Esperar al label que contiene el texto 'BibTeX'
        bibtex_input = driver.find_element(By.XPATH, '//label[.//span[normalize-space()="BibTeX"]]/input')
        driver.execute_script("""
            arguments[0].checked = true;
            arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
        """, bibtex_input)
        time.sleep(2)

        print("Seleccionando formato de citaciones...")
        # Esperar y clic en el label que contiene el texto 'Citations & Abstract'
        citation_input = driver.find_element(
            By.XPATH, '//label[.//span[normalize-space()="Citation and Abstract"]]/input'
        )
        driver.execute_script("""
            arguments[0].checked = true;
            arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
        """, citation_input)
        time.sleep(4)

        print("Clickeando 'Descargar'...")
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.stats-SearchResults_Citation_Download.xpl-btn-primary"))
        )
        driver.execute_script("arguments[0].click();", download_button)
        print("Descarga realizada")

        time.sleep(7)

        # Mover el archivo .bib descargado
        bib_files = sorted(Path(download_dir).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not bib_files:
            print("No hay archivo .bib encontrado en descargas")
            return

        latest_file = bib_files[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = output_dir / f"ieee_scraped_{timestamp}.bib"
        shutil.move(str(latest_file), final_path)

        print(f"BibTeX guardado en: {final_path}")

    except Exception as e:
        print(f"Error en el scraping de página {i}: {e}")
    finally:
        driver.quit()    

def configure_firefox(firefox_options, download_dir):
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.dir", download_dir)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain, application/x-bibtex")
    firefox_options.set_preference("pdfjs.disabled", True)

if __name__ == "__main__":
    try:
        start = int(input("Página de Inicio (e.g. 1): "))
        count = int(input("¿Cuántas páginas para el scrape?: "))

        # Recorrer las páginas
        for i in range(start, start + count):
            print(f"\n>>> Scraping página {i}")
            scrape_ieee_bibtex(i)

    except ValueError:
        print("Invalid input")