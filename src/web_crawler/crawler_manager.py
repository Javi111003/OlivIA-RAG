from urllib.parse import urljoin, urlparse, unquote
from queue import Queue
from bs4 import BeautifulSoup
import requests
import os
import json
from headers import HEADERS
from content_parser import extract_content_from_url

MAX_DEPTH_PER_URL = 1
INVALID_LINKS = ["youtube.com", "facebook.com", "twitter.com", "instagram.com", "tiktok.com"]
JSON_FILE = os.path.join(os.path.dirname(__file__), "crawler_seed.json")

def load_links():
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Error: El archivo JSON tiene un formato inválido.")
        return {}
    except FileNotFoundError:
        print("Error: Archivo no encontrado.")
        return {}

class WebCrawler():
    def __init__(self, subject: str):
        self.subject = subject
        self.seed = load_links()
        self.links = self.craw_web()

    def _init_url_queue(self) -> Queue:
        url_queue = Queue()
        for url in self.seed[self.subject]:
            url_queue.put((url, 0))
        return url_queue
    
    def _is_spanish_page(self ,soup):
        html_tag = soup.find("html")
        if html_tag and html_tag.has_attr("lang"):
            return html_tag["lang"].lower().startswith("es")
        return False
    
    def _is_valid_link(self, url : str) -> bool:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ["http", "https"]:
            return False
        decoded_path = unquote(parsed_url.path)
        decoded_query = unquote(parsed_url.query)
        url_for_keyword_check = (decoded_path + "?" + decoded_query).lower()
        for domain in INVALID_LINKS:
            if domain in url_for_keyword_check:
                return False
        invalid_keywords_url = ["login", "register", "cart", "checkout", "privacy-policy", "terms-of-service", "sitemap", "feedback", "archive", "tag/", "category/", "comment", "forum", "javascript:", "mailto:", "tel:"]
        for keyword in invalid_keywords_url:
            if keyword in url_for_keyword_check:
                return False 
        invalid_extensions = [".doc", ".xls", ".ppt", ".zip", ".rar", ".mp4", ".mp3", ".jpg", ".png", ".gif", ".css", ".js"] 
        for ext in invalid_extensions:
            if url_for_keyword_check.endswith(ext):
                return False
            if f".{ext.lstrip('.')}/" in url_for_keyword_check:
                return False    
        math_vocabulary = ["álgebra", "cálculo", "geometría", "estadística", "teorema", "ecuación", "fórmula", "matemáticas", "problemas", "complejos", "números", "demostración"]
        hist_vocabulary = ["historia", "cuba", "revolucion", "independencia", "batalla", "colonia", "cubana", "partido", "Baraguá"]
        esp_lit_vocabulary = ["literatura", "gramática", "poesía", "novela", "cuento", "autor", "escritor", "ortografía", "sintaxis", "verbos", "adjetivos", "sustantivos", "sinónimos"]
        current_vocabulary = []
        if self.subject == "math":
            current_vocabulary = math_vocabulary
        elif self.subject == "esp-lit":
            current_vocabulary = esp_lit_vocabulary
        elif self.subject == "hist":
            current_vocabulary = hist_vocabulary
        else:
            return False
        is_relevant = False
        for keyword in current_vocabulary:
            if keyword in url_for_keyword_check:
                is_relevant = True
                break
        if not is_relevant:
            return False
    
    def craw_web(self) -> set:
        url_queue = self._init_url_queue()
        visited = set()
        while not url_queue.empty():
            url, depth = url_queue.get()
            if url in visited or depth == MAX_DEPTH_PER_URL:
                continue
            visited.add(url)
            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[ERROR] No se pudo acceder a {url} -> {e}")
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            if not self._is_spanish_page(soup):
                continue
            base_url = response.url  
            for tag in soup.find_all('a', href=True):
                href = urljoin(base_url, tag['href'])
                if href in visited :
                    continue
                if self._is_valid_link(href.lower()):
                    url_queue.put((href, depth + 1))      
        return visited
    
def main():
    # Puedes cambiar esto por cualquier tema
    subject = input("Introduce la materia de la buscaras: ")
    #query = input("Introduce ahora la consulta: ")

    crawler = WebCrawler(subject)

    print("\n🔗 Enlaces encontrados relacionados con la consulta:\n")
    for i, link in enumerate(crawler.links, 1):
        print(f"{i}. {link}")
    
    path_for_save_data = f".data/raw/{subject}/"
    path_for_save_data = f".data/raw/{subject}"
    for link in crawler.links:
        data = extract_content_from_url(link, subject)
        if not data: continue
        # Asegura que el directorio existe
        os.makedirs(path_for_save_data, exist_ok=True)

        # Crea nombre de archivo usando el título (limpiando caracteres no válidos)
        titulo = data["title"].replace("/", "_").replace("\\", "_")
        ruta_archivo = os.path.join(path_for_save_data, f"{titulo}.txt")

        # Escribe el contenido como JSON
        with open(ruta_archivo, "w", encoding="utf-8") as archivo:
            json.dump(data, archivo, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()