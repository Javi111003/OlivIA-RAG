# config.py
import os

# Rutas base
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, '.data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'chunks')
EMBEDDINGS_DIR = os.path.join(PROCESSED_DATA_DIR, 'embeddings')
KG_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'kg_data')

# Crea los directorios si no existen
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(KG_DATA_DIR, exist_ok=True)


# Configuración del OCR (para PDFs escaneados)
RUTA_TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #Ejemplo Windows
#RUTA_TESSERACT_CMD = '/usr/local/bin/tesseract' # Ejemplo macOS/Linux (ajusta a tu instalación)

# Otros parámetros de configuración
LANGUAGE = 'es'  # Idioma para el procesamiento de texto
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50