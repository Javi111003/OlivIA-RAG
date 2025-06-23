from  embedding_models.embedding_generator import EmbeddingGenerator
from  generator.llm_provider import MistralLLMProvider
from  data_preparation.text_cleaner import TextCleaner
import numpy as np

class QueryProcessor:
    def __init__(self, query: str , llm: MistralLLMProvider):
        self.query = query
        self.embeding_generator = EmbeddingGenerator()
        self.text_cleaner = TextCleaner()
        self.llm = llm

    def process(self) -> np.ndarray:
        """
        Process the query to ensure it is in a suitable format for retrieval.
        This can include normalization, tokenization, etc.
        """
        cleaned_query = self._clean_query(self.query, lemmatize=True)
        expanded_query = self._expand_query(cleaned_query)
        processed_query = self._generate_query_embedding(expanded_query)
        if not processed_query.any():
            print(f"Warning: Processed query is empty for original query: '{self.query}'")
            return []
        return processed_query
    def _expand_query(self, query: str) -> str:
        """
        Expand the query to include synonyms or related terms.
        This is a placeholder for more complex logic.
        """
        messages = {
            "role": "system",
            "content": f"""Eres un asistente útil que reescribe o expande las consultas de búsqueda para mejorar la recuperación de información.
            Tu objetivo es generar una versión más detallada o con sinónimos de la consulta original, que ayude a encontrar documentos más relevantes en una base de datos.
            Asegúrate de mantener el significado original y la intención de la consulta.
            Ofrece varias palabras clave o frases relevantes separadas por comas, devuelvelo seguido del texto "Consulta expandida: ".
            Si la consulta ya es suficientemente clara, simplemente devuélvela sin cambios.

            Consulta original: "{query}"
            Consulta expandida: [CONSULTA EXPANDIDA AQUÍ]
            """
        }
        response = self.llm.chat_completion(messages,stream=False)
        response = response.strip().replace("Consulta expandida: ", "")
        return response.strip() if response else query  # Fallback to original query if expansion fails
    
    def _clean_query(self, query: str, lemmatize: bool = False) -> str:
        """
        Clean the query to remove any unwanted characters or formatting.
        This is a placeholder for more complex cleaning logic.
        """       
        cleaned_query = self.text_cleaner.clean_element_content(query, "NarrativeText", apply_lemmatization=lemmatize)
        return cleaned_query
    
    def _generate_query_embedding(self, processed_query: str) -> np.ndarray:
        """
        Generate an embedding for the processed query.
        This method uses the EmbeddingGenerator to create an embedding for the query.
        """
        try:
            query_embedding = self.embeding_generator.generate_embedding(processed_query)
            return query_embedding
        except Exception as e:
            print(f"Error generating embedding for query '{processed_query}': {e}")
            return []
        