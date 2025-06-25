from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

        
class KnowledgeArea(BaseModel):
    """Área de conocimiento matemático con puntuación"""
    name: str = Field(description="Nombre del área de conocimiento")
    score: int = Field(ge=0, le=10, description="Puntuación de dominio (0-10)")
    difficulty: int = Field(ge=0, le=10, description="Puntuación de dificultad (0-10)", default_factory=5)
    weight: int = Field(ge=0, le=10, description="Peso para el examen(0-10)", default_factory= 5)
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence_level: Literal["baja", "media", "alta"] = Field(default="media")
    topics_mastered: List[str] = Field(default_factory=list, description="Temas específicos dominados")
    topics_struggling: List[str] = Field(default_factory=list, description="Temas con dificultades")
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: int) -> int:
        return max(0, min(10, v))

class PreUniversityMathKnowledge(BaseModel):
    """Conocimiento matemático preuniversitario estructurado por áreas"""
    
    # Álgebra y Aritmética
    aritmetica_basica: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Aritmética Básica",
            score=5,
            difficulty=3,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    algebra_elemental: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Álgebra Elemental",
            score=5,
            difficulty=5,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    ecuaciones_lineales: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Ecuaciones Lineales",
            score=5,
            difficulty=2,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    sistemas_ecuaciones: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Sistemas de Ecuaciones",
            score=5,
            difficulty=5,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    ecuaciones_cuadraticas: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Ecuaciones Cuadráticas",
            score=5,
            difficulty=7,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    # Geometría
    geometria_plana: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Geometría Plana",
            score=5,
            difficulty=9,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    geometria_espacial: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Geometría Espacial",
            score=5,
            difficulty=8,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    geometria_analitica: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Geometría Analítica",
            score=5,
            difficulty=9,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    # Funciones y Análisis
    funciones_basicas: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Funciones Básicas",
            score=5,
            difficulty=2,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    funciones_cuadraticas: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Funciones Cuadráticas",
            score=5,
            difficulty=4,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    funciones_exponenciales: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Funciones Exponenciales",
            score=5,
            difficulty=4,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    funciones_logaritmicas: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Funciones Logarítmicas",
            score=5,
            difficulty=5,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    # Trigonometría
    trigonometria_basica: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Trigonometría Básica",
            score=5,
            difficulty=6,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    identidades_trigonometricas: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Identidades Trigonométricas",
            score=5,
            difficulty=5,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    # Probabilidad y Estadística
    estadistica_descriptiva: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Estadística Descriptiva",
            score=5,
            difficulty=4,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    probabilidad_basica: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Probabilidad Básica",
            score=5,
            difficulty=4,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    # Cálculo Preuniversitario
    limites_continuidad: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Límites y Continuidad",
            score=5,
            difficulty=9,
            weight=1,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    derivadas_basicas: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Derivadas Básicas",
            score=5,
            difficulty=7,
            weight=1,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    # Conjuntos y Lógica
    teoria_conjuntos: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Teoría de Conjuntos",
            score=5,
            difficulty=6,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    logica_matematica: KnowledgeArea = Field(
        default_factory=lambda: KnowledgeArea(
            name="Lógica Matemática",
            score=5,
            difficulty=8,
            weight=5,
            topics_mastered=[],
            topics_struggling=[]
        )
    )
    
    def get_all_areas(self) -> Dict[str, KnowledgeArea]:
        """Obtiene todas las áreas de conocimiento"""
        return {
            field_name: getattr(self, field_name) 
            for field_name in self.__fields__.keys()
        }
    
    def get_weak_areas(self, threshold: int = 4) -> List[KnowledgeArea]:
        """Obtiene áreas con puntuación baja"""
        return [
            area for area in self.get_all_areas().values()
            if area.score <= threshold
        ]
    
    def get_strong_areas(self, threshold: int = 7) -> List[KnowledgeArea]:
        """Obtiene áreas con puntuación alta"""
        return [
            area for area in self.get_all_areas().values()
            if area.score >= threshold
        ]
    
    def get_overall_score(self) -> float:
        """Calcula puntuación general"""
        areas = self.get_all_areas()
        if not areas:
            return 5.0
        return sum(area.score for area in areas.values()) / len(areas)
    
    def update_area_score(self, area_name: str, new_score: int, confidence: str = "media"):
        """Actualiza la puntuación de un área específica"""
        if hasattr(self, area_name):
            area = getattr(self, area_name)
            area.score = max(0, min(10, new_score))
            area.confidence_level = confidence
            area.last_updated = datetime.now()
class BDIState(BaseModel):
    """Estado BDI (Beliefs, Desires, Intentions) del agente tutor"""
    beliefs: Dict[str, Any] = Field(default_factory=dict, description="Creencias sobre el estudiante")
    desires: List[str] = Field(default_factory=list, description="Objetivos de aprendizaje deseados")
    intentions: Dict[str, Any] = Field(default_factory=dict, description="Plan de acción actual")

class EstudianteProfile(BaseModel):
    """Estado del estudiante para personalización pedagógica"""
    nivel_comprension: Literal["principiante", "intermedio", "avanzado"] = "principiante"
    
    #Nuevo campo para conocimiento matemático preuniversitario
    math_knowledge: PreUniversityMathKnowledge = Field(default_factory=PreUniversityMathKnowledge)
    
    #campos legacy , mantenido por compatibilidad 
    temas_dominados: List[str] = Field(default_factory=list)
    areas_dificultad: List[str] = Field(default_factory=list)
    preferencias_aprendizaje: Dict[str, Any] = Field(default_factory=dict)
    historial_errores: List[str] = Field(default_factory=list)
    
    def sync_legacy_fields(self):
        """Sincroniza campos legacy con el nuevo sistema de knowledge areas"""
        try:
            strong_areas = self.math_knowledge.get_strong_areas()
            self.temas_dominados = [area.name for area in strong_areas]
            
            weak_areas = self.math_knowledge.get_weak_areas()
            self.areas_dificultad = [area.name for area in weak_areas]
            
            # Actualizar nivel de comprensión basado en puntuación general
            overall_score = self.math_knowledge.get_overall_score()
            if overall_score >= 7.5:
                self.nivel_comprension = "avanzado"
            elif overall_score >= 5.5:
                self.nivel_comprension = "intermedio"
            else:
                self.nivel_comprension = "principiante"
                
        except Exception as e:
            # No fallar por errores de sincronización
            import logging
            logging.warning(f"Error en sync_legacy_fields: {e}")   

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
    respuesta_planning: Optional[str] = Field(default=None)
    
    # Control de flujo
    tipo_ayuda_necesaria: Optional[str] = Field(default=None)
    estado_actual: str = Field(default="inicio")
    respuesta_final: Optional[str] = Field(default=None)
    
    # Evaluación y crawler
    calidad_respuesta: Optional[Literal["suficiente", "insuficiente"]] = Field(default=None)
    necesita_crawler: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True
