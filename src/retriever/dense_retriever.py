import numpy as np
from  vector_db.chroma_store import ChromaVectorStore
import json
import os

class DenseRetriever:
    def __init__(self, embeddings_json_path=None, chroma_collection=None):
        """
        embeddings_json_path: str, ruta al archivo JSON con los embeddings y documentos (opcional)
        chroma_collection: objeto de colección ChromaDB (opcional)
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        chroma_db_path = os.path.join(project_root, '.chroma_db')
        self.chromadb = ChromaVectorStore(chroma_db_path)
        if embeddings_json_path:
            with open(embeddings_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.embeddings = np.array([item["embedding"] for item in data])
            self.documents = [item["content"] for item in data]
        elif chroma_collection:
            results = chroma_collection.get(include=['embeddings', 'documents'])
            self.embeddings = np.array(results['embeddings'])
            self.documents = results['documents']
        elif not self.chromadb:
            raise ValueError("Debe proporcionar embeddings_json_path o chroma_collection")

    def retrieve(self, query_embedding, top_k=5):
        """
        query_embedding: np.ndarray of shape (embedding_dim,) o list
        top_k: int, number of top documents to return
        Returns: list of (document, score) tuples
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()
        
        results = self.chromadb.search_similar_chunks(
            query_embedding,
            top_k
        )
        result = []
        for element in results:
            document = element['content']
            score = element['distance']
            result.append((document, score))
        result.sort(key=lambda x: x[1], reverse=False)
        return result[:top_k]
            

        