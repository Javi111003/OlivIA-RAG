from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

class BDIState(BaseModel):
    """Estado BDI (Beliefs, Desires, Intentions) del agente tutor"""
    beliefs: Dict[str, Any] = Field(default_factory=dict, description="Creencias sobre el estudiante")
    desires: List[str] = Field(default_factory=list, description="Objetivos de aprendizaje deseados")
    intentions: Dict[str, Any] = Field(default_factory=dict, description="Plan de acción actual")

class EstudianteProfile(BaseModel):
    """Estado del estudiante para personalización pedagógica"""
    nivel_comprension: Literal["principiante", "intermedio", "avanzado"] = "principiante"
    temas_dominados: List[str] = Field(default_factory=list)
    areas_dificultad: List[str] = Field(default_factory=list)
    preferencias_aprendizaje: Dict[str, Any] = Field(default_factory=dict)
    historial_errores: List[str] = Field(default_factory=list)

class EstadoConversacion(BaseModel):
    """Estado completo de la conversación con contexto BDI"""
    # Datos básicos
    consulta_inicial: str = Field(description="Consulta original del usuario")
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Estado BDI y estudiante
    bdi_state: Optional[BDIState] = Field(default=None)
    estado_estudiante: EstudianteProfile = Field(default_factory=EstudianteProfile)
    
    # Contexto RAG
    contexto_recuperado: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Respuestas de agentes especializados
    respuesta_math_expert: Optional[str] = Field(default=None)
    respuesta_exam_creator: Optional[str] = Field(default=None)
    respuesta_student_simulator: Optional[str] = Field(default=None)
    
    # Control de flujo
    tipo_ayuda_necesaria: Optional[str] = Field(default=None)
    estado_actual: str = Field(default="inicio")
    respuesta_final: Optional[str] = Field(default=None)
    
    # Evaluación y crawler
    calidad_respuesta: Optional[Literal["suficiente", "insuficiente"]] = Field(default=None)
    necesita_crawler: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True