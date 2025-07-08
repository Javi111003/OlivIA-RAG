import sys
from pathlib import Path


# Obtener la ruta al directorio src (asumiendo que C/C1 está dentro de src)
src_path = Path(__file__).parent.parent.parent  # Ajusta según tu estructura
sys.path.append(str(src_path))


import json
from typing import Any, Dict, List
from pydantic import BaseModel
from generator.llm_provider import MistralLLMProvider
from retriever.dense_retriever import DenseRetriever
from retriever.query_processor import QueryProcessor
from embedding_models.embedding_generator import EmbeddingGenerator
from queries_thexp import *


class ChunkLabel(BaseModel):
    is_relevant: bool

def _use_llm_for_evaluation(query: str, document: str, llm: MistralLLMProvider) -> bool:
    llm = llm.with_structured_output(ChunkLabel)
    prompt = f"""
    Pregunta: "{query}"

    ¿Es el siguiente texto relevante para responderla, en un contexto de estructuras de datos y algoritmos?

    Texto: "{document}"

    Responde solo con un JSON que contenga el campo booleano `is_relevant`.
    """
    result: ChunkLabel = llm.invoke(prompt)
    return result.is_relevant

def evaluate_query(query_embedding,
                   query: str, 
                   retriever: DenseRetriever, 
                   llm: MistralLLMProvider) -> Dict[str, Any]:
    chunk_score = []
    doc_scores = retriever.retrieve(query_embedding, top_k= 10)
    for doc, score in doc_scores:
        chunk_text = doc
        is_relevant = _use_llm_for_evaluation(query_embedding, chunk_text, llm)
        chunk_score.append({
            "text": chunk_text,
            "score": score,
            "is_relevant": is_relevant
        })
    return {
        "query": query,
        "scores": chunk_score
    }

def run_experiments(queries: list[str]) -> list[Dict[str, Any]]:
    retriever = DenseRetriever()
    llm = MistralLLMProvider()
    gen = EmbeddingGenerator()
    results = []
    for query in queries:
        query_embedding = gen.generate_embedding(query)
        result = evaluate_query(query_embedding, query, retriever, llm)
        results.append(result)
    return results

def save_results(
    results: List[Dict[str, Any]],
    filepath: str
    ):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[✅] Resultados guardados en {filepath}")

def run_threshold_experiment(
    queries: List[str] = queries_ste,
    output_path: str = "./src/experiments/similarity_threshold/results.json",
    ):
    results = run_experiments(queries)
    save_results(results, output_path)