# src/pipeline.py

import os
import json
import sys

# Añadir el directorio raíz del proyecto al sys.path para importaciones
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_preparation.document_loader import DocumentLoader
from src.data_preparation.text_cleaner import TextCleaner
from src.data_preparation.chunker import SemanticChunker
from src.embedding_models.embedding_generator import EmbeddingGenerator
from src.vector_db.chroma_store import ChromaVectorStore

def run_ingestion_pipeline():
    """
    Ejecuta el pipeline completo desde la carga de documentos hasta la generación
    de embeddings y su almacenamiento en la base de datos vectorial.
    """
    
    # 1. Rutas de directorios
    RAW_DATA_ROOT = os.path.join(project_root, '.data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(project_root, '.data', 'processed')
    CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'chunks')
    EMBEDDINGS_DIR = os.path.join(PROCESSED_DATA_DIR, 'embeddings')
    CHROMA_DB_PATH = os.path.join(project_root, '.chroma_db')

    # Crear directorios si no existen
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # 2. Cargar documentos
    loader = DocumentLoader(base_data_dir=RAW_DATA_ROOT)
    all_extracted_elements = loader.load_all_documents_from_base_dir()

    if not all_extracted_elements:
        print("No se extrajeron elementos de los documentos. Abortando pipeline.")
        return

    # 3. Limpiar texto
    cleaner = TextCleaner()
    # Aplicar lematización: True para lematizar, False para solo limpiar
    cleaned_elements = cleaner.clean_documents(all_extracted_elements, apply_lemmatization=False)
    
    # 4. Chunking Semántico
    chunker = SemanticChunker()
    chunks = chunker.chunk_documents(cleaned_elements)

    # 5. Generación de Embeddings
    embedding_generator = EmbeddingGenerator()
    chunks_with_embeddings = embedding_generator.generate_embeddings_for_chunks(chunks)

    # Guardar chunks con embeddings
    embeddings_output_path = os.path.join(EMBEDDINGS_DIR, 'chunks_with_embeddings.json')
    with open(embeddings_output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
    print(f"Chunks con embeddings guardados en: {embeddings_output_path}")

    # 6. Almacenar en ChromaDB
    vector_store = ChromaVectorStore(path=CHROMA_DB_PATH)
    vector_store.add_chunks(chunks_with_embeddings)
    print(f"Total de chunks en ChromaDB: {vector_store.count_chunks()}")
    
    print("\nPipeline de ingesta completado exitosamente.")

if __name__ == "__main__":
    run_ingestion_pipeline()