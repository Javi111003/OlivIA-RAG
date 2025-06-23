# src/vector_db/chroma_store.py

import chromadb
from typing import List, Dict, Any
import os
import uuid
import numpy as np

class ChromaVectorStore:
    """
    Gestiona la base de datos vectorial utilizando ChromaDB.
    Almacena chunks de texto y sus embeddings, y permite búsquedas de similitud.
    """
    def __init__(self, path: str = "./chroma_db", collection_name: str = "oliv_ia_chunks"):
        """
        Inicializa la base de datos ChromaDB.
        :param path: Ruta donde se almacenarán los datos de ChromaDB.
        :param collection_name: Nombre de la colección (tabla) para los chunks.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        print(f"ChromaDB inicializado. Base de datos en: {path}, Colección: {collection_name}")

    def _get_or_create_collection(self):
        """
        Obtiene una colección existente o crea una nueva.
        Usaremos embedding_function=None porque los embeddings ya vienen precalculados.
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None # Los embeddings serán proporcionados directamente
        )

    def add_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """
        Añade chunks con sus embeddings a la base de datos vectorial.
        
        :param chunks_with_embeddings: Lista de diccionarios, cada uno con 'content',
                                       'metadata' y 'embedding'.
        """
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        existing_ids = set()
        try:
            # Intentar obtener todos los IDs existentes (puede ser lento para colecciones muy grandes)
            # Para una producción a gran escala, se optimizaría este paso
            all_existing = self.collection.get(ids=self.collection.get()['ids'])
            existing_ids.update(all_existing['ids'])
        except Exception as e:
            print(f"Advertencia: No se pudieron obtener los IDs existentes de la colección. {e}")


        for chunk in chunks_with_embeddings:
            chunk_id = chunk['metadata'].get('chunk_id')
            
            # Generar un ID si no existe o si está vacío
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
                chunk['metadata']['chunk_id'] = chunk_id
                print(f"Advertencia: Chunk sin 'chunk_id' en metadata. Se generó uno nuevo: {chunk_id}")
            elif chunk_id in existing_ids:
                # Si el ID ya existe, podemos actualizarlo o saltarlo.
                # Para este pipeline de ingesta, asumiremos que no queremos duplicados
                # y generaremos un nuevo ID si ya existe, o podrías implementar una lógica de "upsert".
                print(f"Advertencia: Chunk con ID '{chunk_id}' ya existe. Generando nuevo ID para evitar duplicados.")
                chunk_id = str(uuid.uuid4())
                chunk['metadata']['chunk_id'] = chunk_id

            if not chunk.get('embedding'):
                print(f"Advertencia: Chunk {chunk_id} no tiene embedding. Saltando este chunk.")
                continue

            ids.append(chunk_id)
            documents.append(chunk['content']) # Almacenar el contenido original o limpio
            
            # Asegúrate de que los metadatos sean serializables (ej. no objetos complejos)
            # ChromaDB almacena los metadatos como JSON.
            # Convertimos page_numbers a lista de ints si es necesario, o asegurarnos que ya lo sea
            metadata_to_store = chunk['metadata'].copy()
            metadata_to_store = chunk['metadata'].copy()
            for key, value in metadata_to_store.items():
                if isinstance(value, list):
                    # Convierte la lista a string, por ejemplo: "1,2,3"
                    metadata_to_store[key] = ",".join(map(str, value))
            metadatas.append(metadata_to_store) 
            embeddings.append(chunk['embedding'])

        if ids:
            print(f"Añadiendo {len(ids)} chunks a la colección '{self.collection_name}'.")
            # Usa 'upsert' si quieres actualizar documentos existentes con el mismo ID
            # o 'add' si siempre esperas IDs nuevos. 'add' fallará si un ID ya existe.
            # Si generamos nuevos IDs para duplicados, 'add' está bien.
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Chunks añadidos exitosamente.")
        else:
            print("No hay chunks con embeddings válidos para añadir.")

    def search_similar_chunks(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Busca los chunks más similares a un embedding de consulta.

        :param query_embedding: El embedding de la consulta.
        :param n_results: Número de resultados más similares a devolver.
        :return: Lista de diccionarios de chunks similares, con sus distancias.
        """
        if not query_embedding.any():
            print("Advertencia: Embedding de consulta vacío.")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Obtener el documento, sus metadatos y la distancia
        )

        # Formatear los resultados para que sean más fáciles de usar
        formatted_results = []
        if results and results['ids'] and results['ids'][0]: # Asegurarse de que haya resultados válidos
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        return formatted_results

    def count_chunks(self) -> int:
        """Devuelve el número total de chunks en la colección."""
        return self.collection.count()

    def reset_collection(self):
        """Elimina y recrea la colección (útil para desarrollo/limpiar)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Colección '{self.collection_name}' eliminada.")
        except Exception as e:
            print(f"Error al eliminar la colección '{self.collection_name}': {e}")
        self.collection = self._get_or_create_collection()
        print(f"Colección '{self.collection_name}' recreada.")

# Ejemplo de uso (ejecuta este bloque para probar la base de datos vectorial)
if __name__ == "__main__":
    from  embedding_models.embedding_generator import EmbeddingGenerator
    import json
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    chunks_with_embeddings_path = os.path.join(project_root, '.data', 'processed', 'embeddings','chunks_with_embeddings.json')
    chroma_db_path = os.path.join(project_root, '.chroma_db')

    chunks_with_embeddings = []
    if not os.path.exists(chunks_with_embeddings_path):
       print(f"Error: {chunks_with_embeddings_path} no encontrado.")
       print("Por favor, ejecuta 'src/embeddings/embedding_generator.py' primero para generar el archivo.")
       # Generar datos de prueba si no hay un archivo existente
       generator = EmbeddingGenerator()
       sample_chunks_for_test = [
           {"content": "la historia de cuba es fascinante, con muchos eventos importantes.", "metadata": {"chunk_id": "historia_1", "page_numbers": [1], "chunk_type": "narrative"}, "cleaned_content": "la historia de cuba es fascinante, con muchos eventos importantes."},
           {"content": "matematicas avanzadas incluyen calculo diferencial e integral, y algebra lineal.", "metadata": {"chunk_id": "matematica_1", "page_numbers": [1], "chunk_type": "narrative"}, "cleaned_content": "matematicas avanzadas incluyen calculo diferencial e integral, y algebra lineal."},
           {"content": "la revolucion cubana fue un periodo de grandes cambios sociales y politicos.", "metadata": {"chunk_id": "historia_2", "page_numbers": [2], "chunk_type": "narrative"}, "cleaned_content": "la revolucion cubana fue un periodo de grandes cambios sociales y politicos."}
       ]
       chunks_with_embeddings = generator.generate_embeddings_for_chunks(sample_chunks_for_test)
       with open(chunks_with_embeddings_path, 'w', encoding='utf-8') as f:
           json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
       print(f"Chunks de prueba con embeddings generados y guardados en: {chunks_with_embeddings_path}")
    else:
        with open(chunks_with_embeddings_path, 'r', encoding='utf-8') as f:
           chunks_with_embeddings = json.load(f)
        print(f"Cargados {len(chunks_with_embeddings)} chunks con embeddings desde {chunks_with_embeddings_path}.")


    vector_store = ChromaVectorStore(path=chroma_db_path)

    ## Opcional: Si deseas Resetear la colección
    vector_store.reset_collection() 

    vector_store.add_chunks(chunks_with_embeddings)
    print(f"Total de chunks en la DB: {vector_store.count_chunks()}")

    #query_text = "Solucion al ejercicio 1.1 de la 1ra convocatoria del examen de ingreso a la educacion superior 2017-2018"
    #
    #query_embedding_generator = EmbeddingGenerator() 
    #query_embedding = query_embedding_generator.generate_embedding(query_text) 
#
    #print(f"\n--- Búsqueda de chunks similares a: '{query_text}' ---")
    #similar_chunks = vector_store.search_similar_chunks(query_embedding, n_results=3)
#
    #if similar_chunks:
    #    for i, chunk in enumerate(similar_chunks):
    #        print(f"\nResultado {i+1} (Distancia: {chunk['distance']:.4f}):")
    #        print(f"Chunk ID: {chunk['chunk_id']}")
    #        print(f"Content: {chunk['content']}")
    #        print(f"Metadata: {chunk['metadata']}")
    #        print("-" * 30)
    #else:
    #    print("No se encontraron chunks similares.")

    #query_text_math = "problemas de calculo y algebra"
    #query_embedding_math = query_embedding_generator.generate_embedding(query_text_math)
#
    #print(f"\n--- Búsqueda de chunks similares a: '{query_text_math}' ---")
    #similar_chunks_math = vector_store.search_similar_chunks(query_embedding_math, n_results=3)
#
    #if similar_chunks_math:
    #    for i, chunk in enumerate(similar_chunks_math):
    #        print(f"\nResultado {i+1} (Distancia: {chunk['distance']:.4f}):")
    #        print(f"Chunk ID: {chunk['chunk_id']}")
    #        print(f"Content: {chunk['content']}")
    #        print(f"Metadata: {chunk['metadata']}")
    #        print("-" * 30)
    #else:
    #    print("No se encontraron chunks similares.")