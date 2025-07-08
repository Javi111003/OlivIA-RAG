import numpy as np
from typing import List, Set, Tuple

def calculate_precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Calcula la Precisión en el top k (Precision@k).

    Precision@k mide la proporción de documentos relevantes entre los documentos
    recuperados en el top k.

    Args:
        retrieved_doc_ids (List[str]): Una lista de IDs de documentos recuperados por el sistema,
                                        ordenados por similitud descendente. Estos documentos
                                        ya deben estar filtrados por cualquier umbral aplicado.
        relevant_doc_ids (List[str]): Una lista de IDs de documentos que son realmente relevantes
                                      para la consulta (el 'gold standard').
        k (int): El número de documentos superiores (top-k) a considerar para el cálculo.

    Returns:
        float: El valor de Precision@k. Retorna 0.0 si no se recupera ningún documento.
    """
    relevant_doc_ids_set: Set[str] = set(relevant_doc_ids)
    retrieved_at_k_set: Set[str] = set(retrieved_doc_ids[:k])

    tp_at_k: int = len(retrieved_at_k_set.intersection(relevant_doc_ids_set))
    fp_at_k: int = len(retrieved_at_k_set - relevant_doc_ids_set)

    if (tp_at_k + fp_at_k) == 0:
        return 0.0
    return tp_at_k / (tp_at_k + fp_at_k)

def calculate_recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Calcula el Recall en el top k (Recall@k).

    Recall@k mide la proporción de documentos relevantes totales que fueron recuperados
    por el sistema en el top k.

    Args:
        retrieved_doc_ids (List[str]): Una lista de IDs de documentos recuperados por el sistema,
                                        ordenados por similitud descendente. Estos documentos
                                        ya deben estar filtrados por cualquier umbral aplicado.
        relevant_doc_ids (List[str]): Una lista de IDs de documentos que son realmente relevantes
                                      para la consulta (el 'gold standard').
        k (int): El número de documentos superiores (top-k) a considerar para el cálculo.

    Returns:
        float: El valor de Recall@k. Retorna 1.0 si no hay documentos relevantes en el 'gold standard'.
    """
    relevant_doc_ids_set: Set[str] = set(relevant_doc_ids)
    retrieved_at_k_set: Set[str] = set(retrieved_doc_ids[:k])

    tp_at_k: int = len(retrieved_at_k_set.intersection(relevant_doc_ids_set))
    fn_at_k: int = len(relevant_doc_ids_set - retrieved_at_k_set)

    if (tp_at_k + fn_at_k) == 0:
        return 1.0  # No hay relevantes, por lo tanto, el sistema recuperó todo lo que pudo.
    return tp_at_k / (tp_at_k + fn_at_k)

def calculate_f1_at_k(precision_at_k: float, recall_at_k: float) -> float:
    """
    Calcula el F1-score en el top k.

    El F1-score es la media armónica de la Precisión y el Recall, proporcionando
    un equilibrio entre ambas métricas.

    Args:
        precision_at_k (float): El valor de Precision@k.
        recall_at_k (float): El valor de Recall@k.

    Returns:
        float: El valor del F1-score@k. Retorna 0.0 si la suma de precisión y recall es 0.
    """
    if (precision_at_k + recall_at_k) == 0:
        return 0.0
    return 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)