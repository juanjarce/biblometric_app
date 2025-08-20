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

def scrape_acm_bibtex(driver, wait, download_dir, pag_inicio: int, prim_pag, ult_pag):
    """
    Scrapea una página específica de la biblioteca digital de ACM y descarga las citaciones en formato BibTex.
    """
    output_dir = Path("data/raw/ACM")
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ir a la página de ACM
    if prim_pag:
        url = f"https://dl.acm.org/action/doSearch?AllField=generative+artificial+intelligence&startPage={pag_inicio}&pageSize=50"
        driver.get(url)

    try:
        if prim_pag:
            print("Esperando para aceptar las cookies...")
            cookie_button = wait.until(
                EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))
            )
            cookie_button.click()
            print("Cookies aceptadas")

        print("Seleccionando todos los resultados...")
        checkbox = driver.find_element(By.CSS_SELECTOR, "input[name='markall']")
        driver.execute_script("arguments[0].click();", checkbox)
        time.sleep(2)

        print("Abriendo el modal de exportación...")
        export_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.export-citation"))
        )
        driver.execute_script("arguments[0].click();", export_button)
        time.sleep(5) # Tiempo de espera para la carga de la exportación
        if not prim_pag:
            time.sleep(10) # Tiempo de espera para la carga de la exportación (Se demora mas si NO es la primera pagina)

        print("Clickeando 'Descargar citas'...")
        download_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.download__btn[title='Download citation']"))
        )
        driver.execute_script("arguments[0].click();", download_btn)
        print("Descarga realizada")

        time.sleep(5)

        # Mover el archivo .bib descargado
        bib_files = sorted(Path(download_dir).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not bib_files:
            print("No hay archivo .bib encontrado en descargas")
            return

        latest_file = bib_files[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = output_dir / f"acm_scraped_{timestamp}.bib"
        shutil.move(str(latest_file), final_path)

        print(f"BibTeX guardado en: {final_path}")

        # Cerrar el form de exportación
        close_popup = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "i.icon-close_thin"))
        )  
        driver.execute_script("arguments[0].click();", close_popup)
        print("Popup cerrado")
        time.sleep(1)

        # Si no es la última página, ir a la siguiente
        if not ult_pag:
            print("Cambiando a la siguiente página...")
            next_btn = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pagination__btn--next"))
            )
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(3)  # esperar carga de pagina

    except Exception as e:
        print(f"Error en el scraping de página {pag_inicio}: {e}")

def configure_firefox(firefox_options, download_dir):
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.dir", download_dir)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain, application/x-bibtex")
    firefox_options.set_preference("pdfjs.disabled", True)

if __name__ == "__main__":
    try:
        start = int(input("Página de Inicio (e.g. 0): "))
        count = int(input("¿Cuántas páginas para el scrape?: "))

        # Configuración de Firefox
        download_dir = str(Path("downloads").resolve())
        firefox_options = Options()
        configure_firefox(firefox_options, download_dir)

        # Iniciar navegador solo una vez
        driver = webdriver.Firefox(options=firefox_options)
        wait = WebDriverWait(driver, 10)

        # Recorrer las páginas
        for i in range(start, start + count):
            print(f"\n>>> Scraping página {i}")
            scrape_acm_bibtex(driver, wait, download_dir, i, prim_pag=(i == start), ult_pag=(i == start+count-1))
            time.sleep(3)

        # Cerrar navegador solo al final
        input("Enter para cerrar el browser...")
        driver.quit()

    except ValueError:
        print("Invalid input")