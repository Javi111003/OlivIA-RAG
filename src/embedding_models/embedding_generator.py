from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import uuid 

class EmbeddingGenerator:
    """
    Genera embeddings vectoriales para chunks de texto utilizando un modelo pre-entrenado.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicializa el generador de embeddings con un modelo Sentence-BERT.
        :param model_name: Nombre del modelo a cargar desde Hugging Face.
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Modelo de embeddings '{model_name}' cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo de embeddings '{model_name}': {e}")
            self.model = None

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera el embedding para un solo texto.
        :param text: El texto limpio del chunk.
        :return: Un array de numpy representando el embedding.
        """
        if not self.model:
            raise RuntimeError("El modelo de embeddings no se cargó correctamente.")
        
        # Codificar el texto y devolver el array numpy
        embedding = self.model.encode(text, convert_to_tensor=False)
        print(f"Embedding generado para el texto: '{text[:30]}...' (longitud: {len(embedding)})")
        return embedding

    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera embeddings para una lista de chunks.
        Cada chunk debe tener una clave 'content' o 'cleaned_content'.
        Se añadirá una nueva clave 'embedding' al diccionario de cada chunk.

        :param chunks: Lista de diccionarios de chunks (salida de SemanticChunker).
        :return: Lista de diccionarios de chunks, cada uno con un campo 'embedding'.
        """
        if not self.model:
            print("Advertencia: No se generarán embeddings porque el modelo no está cargado.")
            return chunks

        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            text_to_embed = chunk.get('cleaned_content', chunk.get('content', ''))
            
            if 'chunk_id' not in chunk['metadata']:
                chunk['metadata']['chunk_id'] = str(uuid.uuid4())
                print(f"Advertencia: Chunk {i} no tiene 'chunk_id'. Se generó uno: {chunk['metadata']['chunk_id']}")

            if not text_to_embed.strip():
                print(f"Advertencia: Chunk {chunk['metadata']['chunk_id']} tiene contenido vacío. Se asignará embedding vacío.")
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = [] 
            else:
                try:
                    embedding = self.generate_embedding(text_to_embed)
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = embedding
                except Exception as e:
                    print(f"Error al generar embedding para chunk {chunk['metadata']['chunk_id']}: {e}")
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = [] 
            chunks_with_embeddings.append(chunk_with_embedding)
        return chunks_with_embeddings

# Ejemplo de uso
if __name__ == "__main__":
    # Suponiendo que exam_chunks.json contiene la salida deSemanticChunker
    
    # Ruta del archivo de chunks (ajusta según tu estructura de carpetas)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    chunks_file_path = os.path.join(project_root, '.data', 'processed','chunks','exam_chunks.json') 

    sample_chunks = []
    if os.path.exists(chunks_file_path):
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            sample_chunks = json.load(f)
        print(f"Cargados {len(sample_chunks)} chunks desde {chunks_file_path}")
    else:
        print(f"Error: No se encontró el archivo de chunks en {chunks_file_path}")
        print("Creando chunks de prueba manuales para demostración...")
        sample_chunks = [
            {
                "content": "examen de ingreso a la educacion superior curso 2017-2018 1ra convocatoria 1. lee detenidamente y responde.",
                "metadata": {"chunk_id": "test_chunk_1", "page_numbers": [1], "chunk_type": "exam_section", "cleaned_content": "examen de ingreso a la educacion superior curso 2017-2018 1ra convocatoria 1. lee detenidamente y responde."}
            },
            {
                "content": "2. en la figura se muestra una circunferencia de centro o y diametro ab. ademas, se conoce que el punto c pertenece a la circunferencia, el segmento oc es perpendicular al segmento ab y el segmento bc es igual a 6 cm.",
                "metadata": {"chunk_id": "test_chunk_2", "page_numbers": [2], "chunk_type": "exam_exercise", "exercise_id": "2", "cleaned_content": "2. en la figura se muestra una circunferencia de centro o y diametro ab. ademas, se conoce que el punto c pertenece a la circunferencia, el segmento oc es perpendicular al segmento ab y el segmento bc es igual a 6 cm."}
            },
            {
                "content": "Este chunk está vacío.",
                "metadata": {"chunk_id": "test_chunk_3", "page_numbers": [1], "chunk_type": "empty_test", "cleaned_content": ""} # Chunk con contenido vacío para prueba
            }
        ]

    generator = EmbeddingGenerator()
    chunks_with_embeddings = generator.generate_embeddings_for_chunks(sample_chunks)

    print("\n--- Chunks con Embeddings Generados (primeros 3, si hay) ---")
    for i, chunk in enumerate(chunks_with_embeddings[:3]): # Mostrar solo los primeros 3
        print(f"\nCHUNK {i+1} (ID: {chunk['metadata'].get('chunk_id', 'N/A')}):")
        print(f"Content (fragmento): {chunk['content'][:100]}...") # Mostrar un fragmento
        print(f"Embedding length: {len(chunk['embedding'])}")
        if chunk['embedding']: # Solo mostrar si no está vacío
            print(f"Embedding (primeros 5 valores): {chunk['embedding'][:5]}...")
        else:
            print("Embedding: [] (Vacío o error)")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 30)

    output_embeddings_path = os.path.join(project_root, '.data', 'processed', 'embeddings', 'chunks_with_embeddings.json')
    with open(output_embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
    print(f"\nChunks con embeddings guardados en: {output_embeddings_path}")