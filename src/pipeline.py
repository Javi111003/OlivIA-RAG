from retriever.query_processor import QueryProcessor
from retriever.dense_retriever import DenseRetriever
from generator.llm_provider import MistralLLMProvider
from generator.prompt_builder import PromptBuilder
from typing import Any , Dict , Union , Generator
import numpy as np
# from agents.supervisor_agent import SupervisorAgent  # Si tienes agentes

class MainPipeline:
    def __init__(self, llm: MistralLLMProvider, embeddings : list, documents: list[Dict[str,Any]], config = None):
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.documents = documents

    def run(self, user_query: str) -> Union[Generator[str, None, None], str]:
        # 1. Procesar la consulta
        qp = QueryProcessor(user_query, self.llm)
        processed_query = qp.process()

        # 2. Recuperar documentos relevantes
        retriever = DenseRetriever(self.embeddings, self.documents)
        top_docs = retriever.retrieve(processed_query)
        
        builder = PromptBuilder()

        # 3. Construir el prompt para el LLM
        prompt = builder.build_final_response(user_query, context=top_docs)
        messages = {"role": "user", "content": prompt}
        # 4. (Opcional) Pasar el prompt a un agente supervisor
        # supervisor = SupervisorAgent()
        # response = supervisor.handle(prompt)

        # 5. (O simplemente) Obtener respuesta del LLM
        response = self.llm.chat_completion(messages, stream=True)

        return response

# Ejemplo de uso:
if __name__ == "__main__":
    llm = MistralLLMProvider() # Instancia de tu LLM provider
    embeddings = np.ndarray # np.ndarray de embeddings de documentos
    documents = [] # Lista de documentos originales

    pipeline = MainPipeline(llm, embeddings, documents)
    user_query = "¿Qué es la suma?"
    respuesta = pipeline.run(user_query)
    print(respuesta)