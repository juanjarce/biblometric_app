from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from pathlib import Path
from selenium.common.exceptions import TimeoutException
import time
import shutil
from datetime import datetime
import os
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

def scrape_wos(page: int):
    """
    Lanzar una ventana en Firefox browser, navegar a la Libreria Web of Science,
    aceptar las cookies, seleccionar la opcion de descargar todos los resultads, obtenerlas citaciones en formato BibTeX 
    """
    download_dir = str(Path("downloads").resolve())
    output_dir = Path("data/raw/WoS")
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuración de Firefox
    firefox_options = Options()
    configure_firefox(firefox_options, download_dir)

    # Iniciar navegador
    driver = webdriver.Firefox(options=firefox_options)
    wait = WebDriverWait(driver, 15)

    # Ir a la página de Web of Science desde la biblioteca CRAI de la Universidad del quindio
    url = f"https://www-webofscience-com.crai.referencistas.com/wos/woscc/summary/bbe3e96c-ac28-45db-a3ca-7b2c3124b4a2-019d9fb1b6/70fc8305-3be3-4156-8fb2-d12712698ea2-019d9fb188/relevance/{page}"
    driver.get(url)

    try:

        print("Buscando botón de acceso CRAI / Google...")

        acceso_btn = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//a[contains(@href, '/v2/conector/google/solicitar')]"
            ))
        )
        # Scroll por si está fuera de pantalla
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", acceso_btn)
        time.sleep(1)
        acceso_btn.click()
        print("Acceso CRAI / Google clickeado")
        time.sleep(3) # Esperar a que cargue la pagina

        email, password = load_credentials()

        # Acceder a la cuenta CRAI con la info universitaria desde Google
        print("Esperando campo de email...")
        email_input = wait.until(
            EC.presence_of_element_located((By.ID, "identifierId"))
        )
        email_input.clear()
        email_input.send_keys(email)
        print("Email escrito")
        time.sleep(3)

        print("Clickeando botón Continuar...")
        next_button = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//div[@id='identifierNext']//button"
            ))
        )
        next_button.click()
        time.sleep(3) # Esperar a que cargue la pagina

        print("Esperando campo de contraseña...")
        password_input = wait.until(
            EC.presence_of_element_located((By.NAME, "Passwd"))
        )
        password_input.clear()
        password_input.send_keys(password)
        print("Contraseña escrita")
        time.sleep(3)

        print("Clickeando botón Siguiente (password)...")
        next_button = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//div[@id='passwordNext']//button"
            ))
        )
        next_button.click()
        time.sleep(50) # Esperar a que cargue la pagina

        # Aceptar las cookies
        # Click en "Manage cookie preferences"
        print("Aceptando cookies...")
        manage_btn = wait.until(
            EC.element_to_be_clickable((By.ID, "onetrust-pc-btn-handler"))
        )
        manage_btn.click()
        time.sleep(2)

        # Click en "Confirm my choices"
        confirm_btn = wait.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "button.save-preference-btn-handler.onetrust-close-btn-handler"
            ))
        )
        confirm_btn.click()
        time.sleep(1)
        print("Cookies aceptadas correctamente")

        # Pasos para seleccionar los articulos y bajar la informacion en archivos .bib (BibTex)
        print("Esperando checkbox...")
        checkbox = wait.until(
            EC.presence_of_element_located((By.ID, "mat-mdc-checkbox-1-input"))
        )

        driver.execute_script("arguments[0].click();", checkbox)
        print("Checkbox seleccionado (JS)")

        time.sleep(3) # Esperar a que cargue la pagina
        print("Esperando botón Exportar...")
        export_button = wait.until(
            EC.presence_of_element_located((By.ID, "export-trigger-btn"))
        )
        driver.execute_script("""
            const btn = arguments[0];
            btn.scrollIntoView({block: 'center'});

            btn.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
            btn.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
            btn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        """, export_button)
        time.sleep(1)
        print("Botón Exportar clickeado")

        time.sleep(3) # Esperar a que cargue la pagina
        print("Seleccionando opción BibTeX...")
        bibtex_button = wait.until(
            EC.element_to_be_clickable((By.ID, "exportToBibtexButton"))
        )
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", bibtex_button
        )
        time.sleep(0.5)
        bibtex_button.click()
        print("Opción BibTeX seleccionada")
        time.sleep(8)

        print("Abriendo dropdown de opciones...")
        print("Accediendo el ComboBox...")
        actions = ActionChains(driver)
        actions.send_keys(Keys.TAB).pause(0.5) \
            .send_keys(Keys.TAB).pause(0.5) \
            .send_keys(Keys.TAB).pause(0.5) \
            .send_keys(Keys.ENTER).pause(0.5) \
            .send_keys(Keys.ARROW_DOWN).pause(0.4) \
            .perform()
        
        time.sleep(1)
        print("Seleccionando 'Author, Title, Source, Abstract'...")
        option = wait.until(
            EC.presence_of_element_located((
                By.XPATH,
                "//div[@role='menuitem' and @aria-label='Author, Title, Source, Abstract']"
            ))
        )
        driver.execute_script("""
        const el = arguments[0];

        // asegurar visibilidad y foco real
        el.scrollIntoView({block: 'center'});
        el.setAttribute('tabindex', '0');
        el.focus();

        // evento de teclado EXACTO que Angular Material escucha
        el.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Enter',
            code: 'Enter',
            keyCode: 13,
            which: 13,
            bubbles: true
        }));
        el.dispatchEvent(new KeyboardEvent('keyup', {
            key: 'Enter',
            code: 'Enter',
            keyCode: 13,
            which: 13,
            bubbles: true
        }));

        // cortar propagación posterior (Firefox)
        el.blur();
        """, option)
        
        time.sleep(1)
        print("Exportación final...")
        final_export_button = wait.until(
            EC.element_to_be_clickable((By.ID, "exportButton"))
        )
        driver.execute_script("""
        arguments[0].scrollIntoView({block:'center'});
        arguments[0].focus();
        arguments[0].click();
        """, final_export_button)
        print("Exportación iniciada")
        time.sleep(15) # esperar a que se descargue el archivo

        # Guardar el archivo (.bib) descargado en la carpeta indicada
        # Mover el archivo .bib descargado
        bib_files = sorted(
            Path(download_dir).glob("*.bib"),
            key=os.path.getmtime,
            reverse=True
        )

        if not bib_files:
            print("No hay archivo .bib encontrado en descargas")
            return

        latest_file = bib_files[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = output_dir / f"wos_scraped_{timestamp}.bib"

        shutil.move(str(latest_file), final_path)

        print(f"BibTeX guardado en: {final_path}")


    except Exception as e:
        print(f"Error página {page}: {e}")

    finally:
        driver.quit()


def configure_firefox(options, download_dir):
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.dir", download_dir)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain,application/x-bibtex")
    options.set_preference("pdfjs.disabled", True)

def load_credentials(path="data/utils/credentials.txt"):
    creds = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v
    return creds["email"], creds["password"]

if __name__ == "__main__":
    start = int(input("Página inicio: "))
    count = int(input("¿Cuántas páginas?: "))

    for i in range(start, start + count):
        print(f"\n--- Scraping página {i} ---")
        scrape_wos(i)
