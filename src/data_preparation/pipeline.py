# src/data_preparation/pipeline.py

import os
import json
from typing import List, Dict, Any
from tqdm import tqdm # Para barras de progreso
from src.data_preparation.document_loader import DocumentLoader
from src.data_preparation.text_cleaner import TextCleaner
import src.data_preparation.config as config # Importa el archivo de configuración

class DataPreparationPipeline:
    """
    Orquesta los pasos de carga, limpieza y pre-procesamiento de documentos.
    """
    def __init__(self):
        self.loader = DocumentLoader(tesseract_cmd_path=config.RUTA_TESSERACT_CMD)
        self.cleaner = TextCleaner(lang_model="es_core_news_sm")
        self.processed_docs_info = [] # Para almacenar metadatos de documentos procesados

    def _get_document_paths(self, subject: str = None) -> List[str]:
        """
        Obtiene una lista de rutas de archivos de los documentos raw.
        Si se especifica un subject, solo busca en esa carpeta.
        """
        base_path = config.RAW_DATA_DIR
        paths = []
        if subject:
            subject_path = os.path.join(base_path, subject)
            if os.path.isdir(subject_path):
                for root, _, files in os.walk(subject_path):
                    for file in files:
                        if file.startswith('.'): # Ignorar archivos ocultos
                            continue
                        paths.append(os.path.join(root, file))
            else:
                print(f"Advertencia: La carpeta para la asignatura '{subject}' no existe en {base_path}")
        else: # Procesar todas las asignaturas
            for subject_dir in os.listdir(base_path):
                subject_path = os.path.join(base_path, subject_dir)
                if os.path.isdir(subject_path):
                    for root, _, files in os.walk(subject_path):
                        for file in files:
                            if file.startswith('.'):
                                continue
                            paths.append(os.path.join(root, file))
        return paths

    def run_pipeline(self, subject: str = None):
        """
        Ejecuta el pipeline de preparación de datos: carga y limpieza.
        Guarda el texto limpio en un archivo JSON en data/processed/chunks/.
        """
        document_paths = self._get_document_paths(subject)
        if not document_paths:
            print("No se encontraron documentos para procesar.")
            return

        print(f"Iniciando procesamiento de {len(document_paths)} documentos...")
        processed_data_output = []

        for doc_path in tqdm(document_paths, desc="Procesando documentos"):
            doc_name = os.path.basename(doc_path)
            doc_id = os.path.splitext(doc_name)[0] # Usar nombre de archivo como ID inicial
            
            # Inferir asignatura del path (ej. data/raw/matematica/ -> matematica)
            try:
                doc_subject = doc_path.split(os.sep)[-2]
            except IndexError:
                doc_subject = "unknown" # Fallback si la estructura de la ruta no es la esperada

            print(f"Cargando y limpiando: {doc_name}")
            raw_text = self.loader.load_document(doc_path)

            if raw_text:
                cleaned_text = self.cleaner.clean_text(raw_text)
                # lemmatized_text = self.cleaner.lemmatize_text(cleaned_text)

                # Almacenar la información del documento limpio
                processed_data_output.append({
                    "document_id": doc_id,
                    "file_name": doc_name,
                    "subject": doc_subject,
                    "cleaned_text": cleaned_text,
                    # "lemmatized_text": lemmatized_text # Si lo usas
                })
                print(f"Documento {doc_name} procesado y listo para chunking.")
            else:
                print(f"No se pudo procesar el documento: {doc_name}. Saltando.")

        # Guardar los textos limpios en un archivo JSON en data/processed/chunks/
        # Aunque el nombre sea 'chunks', por ahora contiene el documento completo limpio.
        # En la siguiente fase, este archivo se dividirá en chunks más pequeños.
        output_file_name = f"cleaned_documents_{subject if subject else 'all'}.json"
        output_path = os.path.join(config.CHUNKS_DIR, output_file_name) # Se guarda en la carpeta de chunks
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data_output, f, ensure_ascii=False, indent=4)
        
        print(f"Procesamiento inicial completado. Textos limpios guardados en: {output_path}")
        self.processed_docs_info = processed_data_output
        return processed_data_output

# Para ejecutar el pipeline desde la línea de comandos
if __name__ == "__main__":
    # Ejemplo: crea un archivo de texto simple y guárdalo como test.txt en esa carpeta para probar
    # O mejor aún, usa tus PDFs reales.

    # Para procesar solo documentos de Matemática
    pipeline = DataPreparationPipeline()
    math_docs = pipeline.run_pipeline(subject="math")

    # Para procesar documentos de todas las asignaturas
    # all_docs = pipeline.run_pipeline(subject=None)

    # Puedes imprimir los primeros 500 caracteres del texto limpio del primer documento para verificar
    if math_docs:
        print("\nPrimeros 500 caracteres del texto limpio del primer documento procesado:")
        print(math_docs[0]['cleaned_text'][:500])
        print(f"Guardado en: {os.path.join(config.CHUNKS_DIR, 'cleaned_documents_matematica.json')}")