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

def scrape_acm_bibtex():
    """
    Lanzar una ventana en Firefox browser, navegar a la Libreria Digital de ACM,
    aceptar las cookies, seleccionar la opcion de descargar todos los resultads, obtenerlas citaciones en formato BibTeX 
    """
    download_dir = str(Path("downloads").resolve())
    output_dir = Path("data/raw/ACM2")
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuración de Firefox
    firefox_options = Options()
    configure_firefox(firefox_options, download_dir)

    # Iniciar navegador
    driver = webdriver.Firefox(options=firefox_options)
    wait = WebDriverWait(driver, 10)

    # Ir a la página de ACM
    url = f"https://dl.acm.org/action/doSearch?AllField=computational+thinking&startPage=0&pageSize=50"
    driver.get(url)

    try:
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
        time.sleep(2)

        print("Clickeando 'All Results'...")
        try:
            all_results_btn = wait.until(
                EC.element_to_be_clickable((By.ID, "allResults"))
            )
            driver.execute_script("arguments[0].click();", all_results_btn)
            print("Botón 'All Results' clickeado")
            time.sleep(5)
        except:
            print("No se encontró el botón 'All Results'")

        print("Clickeando 'Download'...")
        try:
            download_btn = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "downloadBtn"))
            )
            driver.execute_script("arguments[0].click();", download_btn)
            print("Botón 'Download' clickeado")
            time.sleep(2)
        except:
            print("No se encontró el botón 'Download'")

        # Tiempo de espera mientras se prepara el archivo completo
        print("Esperando a 'Download Now'")
        time.sleep(60)

        print("Clickeando 'Download Now'...")
        try:
            close_export_popup = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn.blue.searchCiteExport-popup__close.pull-right"))
            )       
            driver.execute_script("arguments[0].click();", close_export_popup)
            print("Popup de exportación cerrado")
            time.sleep(1)
        except:
            print("No se encontró el botón para cerrar el popup de exportación")

        print("Descarga realizada")

        time.sleep(5)
        bib_files = sorted(Path(download_dir).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not bib_files:
            print("No hay archivo .bib encontrado en descargas")

        latest_file = bib_files[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = output_dir / f"acm_scraped_{timestamp}.bib"
        shutil.move(str(latest_file), final_path)

        print(f"BibTeX guardado en: {final_path}")

        input("Enter para cerrar el navegador...")
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        driver.quit()

def configure_firefox(firefox_options, download_dir):
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.dir", download_dir)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain, application/x-bibtex")
    firefox_options.set_preference("pdfjs.disabled", True)

if __name__ == "__main__":
    try:
        scrape_acm_bibtex()
    except ValueError:
        print("Error descargando todos los archivos")