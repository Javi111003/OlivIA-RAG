from pydantic import BaseModel, field_validator, Field
from typing import Literal, List, Dict

class SupervisorDecision(BaseModel):
    """Decisión estructurada del supervisor"""
    next_agent: Literal["math_expert", "exam_creator", "student_simulator", "evaluator", "FINISH"]
    reasoning: str = Field(description="Razonamiento detrás de la decisión")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza en la decisión (0-1)")
    
    @field_validator('next_agent')
    @classmethod
    def validate_agent(cls, v: str) -> str:
        valid_agents = ["math_expert", "exam_creator", "student_simulator", "evaluator", "FINISH"]
        if v not in valid_agents:
            raise ValueError(f"Agente debe ser uno de: {valid_agents}")
        return v

class MathExpertResponse(BaseModel):
    """Respuesta estructurada del experto en matemáticas"""
    explanation: str = Field(description="Explicación matemática detallada")
    formulas: List[str] = Field(default_factory=list, description="Fórmulas utilizadas")
    difficulty_level: Literal["básico", "intermedio", "avanzado"] = Field(description="Nivel de dificultad")
    related_concepts: List[str] = Field(default_factory=list, description="Conceptos relacionados")
    
    @field_validator('formulas', 'related_concepts')
    @classmethod
    def validate_lists_not_empty_strings(cls, v: List[str]) -> List[str]:
        # Filtrar strings vacíos de las listas
        return [item.strip() for item in v if item.strip()]

class ExamCreatorResponse(BaseModel):
    """Respuesta estructurada del creador de exámenes"""
    exam_title: str = Field(description="Título del examen")
    questions: List[str] = Field(description="Lista de preguntas del examen")
    difficulty_level: Literal["básico", "intermedio", "avanzado"] = Field(description="Nivel de dificultad")
    estimated_time: int = Field(gt=0, description="Tiempo estimado en minutos")
    topic_coverage: List[str] = Field(default_factory=list, description="Temas cubiertos")
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("Debe haber al menos una pregunta")
        return [q.strip() for q in v if q.strip()]

class StudentSimulationResponse(BaseModel):
    """Respuesta estructurada del simulador de estudiante"""
    student_answer: str = Field(description="Respuesta simulada del estudiante")
    confidence_level: Literal["baja", "media", "alta"] = Field(description="Nivel de confianza del estudiante")
    common_mistakes: List[str] = Field(default_factory=list, description="Errores comunes que podría cometer")
    reasoning_process: str = Field(description="Proceso de razonamiento del estudiante")
    areas_of_confusion: List[str] = Field(default_factory=list, description="Áreas donde el estudiante se confunde")
    learning_metrics: Dict[str, int] = Field(default_factory=dict, description="Métricas de aprendizaje (0-100)")
    cognitive_load: Literal["bajo", "medio", "alto"] = Field(description="Carga cognitiva del estudiante")
    motivation_change: int = Field(ge=-10, le=10, description="Cambio en motivación (-10 a +10)")
    predicted_performance: int = Field(ge=0, le=100, description="Rendimiento predicho (0-100)")
    
    @field_validator('learning_metrics')
    @classmethod
    def validate_metrics(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Valida que las métricas estén en rango 0-100"""
        validated = {}
        for key, value in v.items():
            validated[key] = max(0, min(100, value))
        return validated

class ResponseEvaluation(BaseModel):
    """Evaluación estructurada de la respuesta"""
    is_sufficient: bool = Field(description="Si la respuesta es suficiente")
    correctness_score: float = Field(ge=0.0, le=1.0, description="Puntuación de corrección (0-1)")
    clarity_score: float = Field(ge=0.0, le=1.0, description="Puntuación de claridad (0-1)")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Puntuación de completitud (0-1)")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Puntuación de relevancia (0-1)")
    needs_more_context: bool = Field(description="Si necesita más contexto/información")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Sugerencias de mejora")
    overall_quality: Literal["poor", "acceptable", "good", "excellent"] = Field(description="Calidad general")
    
    @field_validator('correctness_score', 'clarity_score', 'completeness_score', 'relevance_score')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        return round(v, 2)

class CrawlerTriggerResponse(BaseModel):
    """Respuesta del disparador de crawler"""
    should_crawl: bool = Field(description="Si debe activarse el crawler")
    search_queries: List[str] = Field(description="Consultas de búsqueda para el crawler")
    priority_level: Literal["baja", "media", "alta"] = Field(description="Prioridad de la búsqueda")
    expected_sources: List[str] = Field(default_factory=list, description="Fuentes esperadas donde buscar")
    
    @field_validator('search_queries')
    @classmethod
    def validate_search_queries(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Debe haber al menos una consulta de búsqueda")
        return [q.strip() for q in v if q.strip()]
    
class KnowledgeAnalysisResponse(BaseModel):
    """Respuesta del análisis de conocimiento"""
    areas_analyzed: List[str] = Field(description="Áreas analizadas")
    knowledge_updates: Dict[str, Dict] = Field(description="Actualizaciones de conocimiento por área")
    overall_assessment: str = Field(description="Evaluación general")
    recommendations: List[str] = Field(description="Recomendaciones específicas")

class PlanningResponse(BaseModel):
    """Respuesta del planificador de estudio"""
    plan: Dict[str, float] = Field(description= "Plan de estudio")
    score: float = Field(description= "score del plan")