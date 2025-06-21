import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever:
    def __init__(self, embeddings, documents):
        """
        embeddings: np.ndarray of shape (num_documents, embedding_dim)
        documents: list of str, original documents
        """
        self.embeddings = embeddings
        self.documents = documents

    def retrieve(self, query_embedding, top_k=5):
        """
        query_embedding: np.ndarray of shape (embedding_dim,)
        top_k: int, number of top documents to return
        Returns: list of (document, score) tuples
        """
        sims = cosine_similarity(
            query_embedding.reshape(1, -1), self.embeddings
        ).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.documents[i], sims[i]) for i in top_indices]