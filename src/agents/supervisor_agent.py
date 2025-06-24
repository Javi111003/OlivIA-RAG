from typing import Any, Dict, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from  agents.dto_s.agent_formated_responses import SupervisorDecision
import asyncio
import logging
from  generator.llm_provider import MistralLLMProvider
from  agents.dto_s.agent_state import BDIState, EstadoConversacion

logger = logging.getLogger(__name__)

class SupervisorAgent:
    
    def __init__(self, llm: MistralLLMProvider):
        self.llm = llm
        self.llm_structured = llm.with_structured_output(SupervisorDecision)
        self.parser = JsonOutputParser(pydantic_object=SupervisorDecision)

        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un tutor supervisor especializado en matemáticas con arquitectura BDI.
            
            CONTEXTO BDI ACTUAL:
            - Creencias sobre el estudiante: {student_beliefs}
            - Objetivos de aprendizaje: {learning_goals}
            - Plan de acción actual: {action_plan}
            
            ESTADO DEL ESTUDIANTE:
            - Nivel: {comprehension_level}
            - Temas dominados: {mastered_topics}
            - Áreas de dificultad: {difficulty_areas}
            - Errores recurrentes: {common_errors}
            
            CONTEXTO DE CONVERSACIÓN:
            - Consulta inicial: {consulta_inicial}
            - Últimas 3 interacciones: {recent_interactions}
            - Contexto recuperado: {contexto_disponible}
            
            ESTADO ACTUAL DEL PIPELINE:
            - Estado actual: {estado_actual}
            - Respuesta math_expert: {respuesta_math_expert}
            - Respuesta exam_creator: {respuesta_exam_creator}
            - Necesita más información: {necesita_crawler}
            
            AGENTES DISPONIBLES:
            - **math_expert**: Explicación matemática profunda y rigurosa
            - **exam_creator**: Crear exámenes, quizzes, evaluaciones y preguntas de práctica
            - **evaluator**: Evaluar comprensión y calidad de respuestas generadas
            - **FINISH**: La consulta está completamente resuelta y lista para entregar
            
            REGLAS DE DECISIÓN:
            1. Si pide crear examen/quiz/evaluación/práctica → usar "exam_creator"
            2. Si NO hay respuesta del exam_creator ni tampoco del math_expert → usar "math_expert"
            3. Si hay respuesta del math_expert pero no se ha evaluado , siempre debe evaluarse → usar "evaluator"
            4. Si hay respuesta del exam_creator pero no se ha evaluado → usar "evaluator"
            5. Si hay respuesta evaluada y es de buena calidad → usar "FINISH"
            6. Si hay problemas de calidad → volver al agente correspondiente
            
            PALABRAS CLAVE PARA EXAM_CREATOR:
            - "crea un examen", "genera un quiz", "haz preguntas"
            - "evaluación", "test", "práctica", "ejercicios"
            - "examen sobre", "quiz de", "preguntas de"
            - "prueba", "assessment", "evaluación"
            
            PALABRAS CLAVE PARA MATH_EXPERT:
            - "explica", "qué es", "cómo resolver", "demuestra"
            - "teorema", "fórmula", "concepto", "definición"
            - "ayúdame a entender", "no entiendo"
            
            Analiza la consulta inicial y decide el agente más apropiado.
            Considera las palabras clave y el contexto de la conversación.
            
            {format_instructions}
            """
        )
    
    def supervisor_router(self, estado: EstadoConversacion) -> str:
        """Router que determina el próximo agente basado en la decisión del supervisor"""
        logger.info(f"Router evaluando: tipo_ayuda={estado.tipo_ayuda_necesaria}, estado={estado.estado_actual}")
        
        # Validar que la decisión sea válida
        opciones_validas = ["math_expert", "exam_creator","evaluator", "FINISH"]
        decision = estado.tipo_ayuda_necesaria or "math_expert"
        
        if decision not in opciones_validas:
            logger.warning(f"Decisión inválida '{decision}', usando 'math_expert' por defecto")
            decision = "math_expert"
        
        logger.info(f"Router decide: {decision}")
        return decision

    async def supervisor_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Cadena principal del supervisor con manejo robusto de errores"""
        try:
            logger.info(f"Supervisor evaluando estado: {estado.estado_actual}")
            
            bdi_context = self.extraer_contexto_bdi(estado)
            student_context = estado.estado_estudiante.model_dump()
            
            # Preparar datos para el prompt
            prompt_data = {
                "student_beliefs": bdi_context.get("beliefs", {}),
                "learning_goals": bdi_context.get("desires", []),
                "action_plan": bdi_context.get("intentions", {}),
                "comprehension_level": student_context.get("nivel_comprension"),
                "mastered_topics": student_context.get("temas_dominados", []),
                "difficulty_areas": student_context.get("areas_dificultad", []),
                "common_errors": student_context.get("historial_errores", []),
                "consulta_inicial": estado.consulta_inicial,
                "recent_interactions": self.obtener_interacciones_recientes(estado, 3),
                "contexto_disponible": estado.contexto_recuperado,
                "estado_actual": estado.estado_actual,
                "respuesta_math_expert": estado.respuesta_math_expert or "No disponible",
                "respuesta_exam_creator": estado.respuesta_exam_creator or "No disponible",
                "necesita_crawler": estado.necesita_crawler,
                "format_instructions": self.parser.get_format_instructions()
            }
            
            # Ejecutar decisión del LLM con fallback
            try:
                formatted_prompt = self.prompt.format(**prompt_data)
                respuesta = await self.llm_structured.ainvoke(formatted_prompt)
                
                # Validar respuesta estructurada
                if isinstance(respuesta, dict):
                    respuesta = SupervisorDecision(
                        next_agent=respuesta.get('next_agent', 'math_expert'),
                        reasoning=respuesta.get('reasoning', 'Decisión automática'),
                        confidence=respuesta.get('confidence', 0.8)
                    )
                    
            except Exception as structured_error:
                logger.warning(f"Structured output falló: {structured_error}, usando parser")
                try:
                    formatted_prompt = self.prompt.format(**prompt_data)
                    raw_response = await self.llm.ainvoke(formatted_prompt)
                    respuesta_dict = self.parser.parse(raw_response)
                    
                    if isinstance(respuesta_dict, dict):
                        respuesta = SupervisorDecision(
                            next_agent=respuesta_dict.get('next_agent', 'math_expert'),
                            reasoning=respuesta_dict.get('reasoning', 'Decisión automática'),
                            confidence=respuesta_dict.get('confidence', 0.8)
                        )
                    else:
                        respuesta = respuesta_dict
                        
                except Exception as parse_error:
                    logger.error(f"Error parseando supervisor: {parse_error}")
                    # Fallback inteligente basado en el estado actual
                    respuesta = self._crear_decision_fallback(estado)
            
            opciones_validas = ["math_expert", "exam_creator", "evaluator", "FINISH"]
            if respuesta.next_agent not in opciones_validas:
                logger.warning(f"Agente inválido '{respuesta.next_agent}', usando fallback")
                respuesta = self._crear_decision_fallback(estado)
            
            estado.tipo_ayuda_necesaria = respuesta.next_agent
            estado.estado_actual = f"supervisor_decidio_{respuesta.next_agent}"
            
            self.actualizar_bdi_state(estado, respuesta)
            
            logger.info(f"Supervisor decidió: {respuesta.next_agent} (confianza: {respuesta.confidence})")
            logger.info(f"Razonamiento: {respuesta.reasoning}")
            return estado
            
        except Exception as e:
            logger.error(f"Error crítico en supervisor: {e}")
            # Fallback de emergencia
            fallback_decision = self._crear_decision_fallback(estado)
            estado.tipo_ayuda_necesaria = fallback_decision.next_agent
            estado.estado_actual = f"supervisor_fallback_{fallback_decision.next_agent}"
            return estado
    
    def _crear_decision_fallback(self, estado: EstadoConversacion) -> SupervisorDecision:
        """Crea una decisión de fallback basada en el estado actual"""
        logger.info(f"🔍 Evaluando fallback - Estado: {estado.estado_actual}")
        logger.info(f"🔍 Respuesta math_expert: {bool(estado.respuesta_math_expert)}")
        logger.info(f"🔍 Respuesta exam_creator: {bool(estado.respuesta_exam_creator)}")
        
        consulta_lower = estado.consulta_inicial.lower()
        
        # Palabras clave para exam_creator
        palabras_examen = ["examen", "quiz", "test", "evaluación", "preguntas", "práctica", 
                        "ejercicios", "crea", "genera", "haz un", "prueba", "assessment"]
        
        # Palabras clave para math_expert  
        palabras_matematicas = ["explica", "qué es", "cómo", "teorema", "fórmula", 
                            "concepto", "definición", "resolver", "demuestra", "solucion", "solución"]
        
        if any(palabra in consulta_lower for palabra in palabras_examen) and not estado.respuesta_exam_creator:
            return SupervisorDecision(
                next_agent="exam_creator",
                reasoning="La consulta solicita crear un examen o evaluación",
                confidence=0.9
            )
        
        if any(palabra in consulta_lower for palabra in palabras_matematicas) and not estado.respuesta_math_expert and not estado.respuesta_exam_creator:
            return SupervisorDecision(
                next_agent="math_expert",
                reasoning="La consulta solicita explicación matemática y no hay respuesta aún",
                confidence=0.9
            )
        
        if (estado.respuesta_exam_creator and 
            estado.estado_actual not in ["evaluator_completado", "exam_creator_evaluado"]):
            return SupervisorDecision(
                next_agent="evaluator",
                reasoning="Hay respuesta del exam_creator, necesita evaluación única",
                confidence=0.8
            )
        
        if (estado.respuesta_math_expert and 
            estado.estado_actual not in ["evaluator_completado", "math_expert_evaluado"]):
            return SupervisorDecision(
                next_agent="evaluator",
                reasoning="Hay respuesta del math_expert, necesita evaluación única",
                confidence=0.8
            )
        
        if estado.estado_actual in ["evaluator_completado", "math_expert_evaluado", "exam_creator_evaluado"]:
            return SupervisorDecision(
                next_agent="FINISH",
                reasoning="Ya se evaluó la respuesta, proceso completado",
                confidence=0.9
            )
        
        if (estado.respuesta_math_expert or estado.respuesta_exam_creator):
            if len(estado.chat_history) > 0:
                ultima_consulta = estado.chat_history[-1].get("content", "")
                if estado.consulta_inicial != ultima_consulta:
                    if any(palabra in consulta_lower for palabra in palabras_examen):
                        return SupervisorDecision(
                            next_agent="exam_creator",
                            reasoning="Nueva consulta sobre examen",
                            confidence=0.8
                        )
                    else:
                        return SupervisorDecision(
                            next_agent="math_expert", 
                            reasoning="Nueva consulta matemática",
                            confidence=0.8
                        )
            
            return SupervisorDecision(
                next_agent="FINISH",
                reasoning="Ya hay respuesta disponible, proceso completado",
                confidence=0.9
            )
        
        # Fallback por defecto: math_expert si no hay nada
        return SupervisorDecision(
            next_agent="math_expert",
            reasoning="Sin respuestas disponibles, usar math_expert por defecto",
            confidence=0.7
        )
    
    def extraer_contexto_bdi(self, estado: EstadoConversacion) -> Dict[str, Any]:
        """Extrae y estructura el contexto BDI"""
        if not estado.bdi_state:
            return {"beliefs": {}, "desires": [], "intentions": {}}
        
        return {
            "beliefs": estado.bdi_state.beliefs,
            "desires": estado.bdi_state.desires,
            "intentions": estado.bdi_state.intentions
        }

    def obtener_interacciones_recientes(self, estado: EstadoConversacion, n: int = 3) -> List[Dict]:
        """Obtiene las últimas n interacciones relevantes"""
        return estado.chat_history[-n:] if len(estado.chat_history) >= n else estado.chat_history

    def actualizar_bdi_state(self, estado: EstadoConversacion, decision: SupervisorDecision) -> None:
        """Actualiza el estado BDI basado en la decisión del supervisor"""
        if not estado.bdi_state:
            estado.bdi_state = BDIState()
        
        # Actualizar creencias sobre el estudiante
        estado.bdi_state.beliefs["last_decision"] = decision.next_agent
        estado.bdi_state.beliefs["decision_confidence"] = decision.confidence
        estado.bdi_state.beliefs["reasoning"] = decision.reasoning
        
        # Actualizar intenciones
        estado.bdi_state.intentions["current_action"] = decision.next_agent
        estado.bdi_state.intentions["expected_outcome"] = f"Ejecutar {decision.next_agent}"

def decidir_fallback(self, consulta: str) -> str:
    """Decisión de fallback basada en análisis simple de palabras clave"""
    consulta_lower = consulta.lower()
    
    if any(word in consulta_lower for word in ["examen", "quiz", "test", "evaluación", "preguntas", "crea", "genera", "haz un"]):
        return "exam_creator"
    elif any(word in consulta_lower for word in ["teorema", "demostración", "concepto", "definición", "explica", "qué es"]):
        return "math_expert"
    elif any(word in consulta_lower for word in ["evaluar", "revisar", "calidad"]):
        return "evaluator"
    else:
        return "math_expert"  # Default seguro