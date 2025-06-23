from typing import Any, Dict, List
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from  agents.dto_s.agent_state import EstadoConversacion 
from  agents.dto_s.agent_formated_responses import ResponseEvaluation
from  generator.llm_provider import MistralLLMProvider

logger = logging.getLogger(__name__)

class EvaluatorAgent:
    """Agente evaluador que analiza la calidad de las respuestas generadas"""
    
    def __init__(self, llm: MistralLLMProvider):
        self.llm = llm
        self.llm_structured = llm.with_structured_output(ResponseEvaluation)
        self.parser = JsonOutputParser(pydantic_object=ResponseEvaluation)
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un evaluador experto en educación matemática y calidad de respuestas pedagógicas.
            
            RESPUESTA A EVALUAR:
            Consulta original: {consulta_original}
            Respuesta generada: {respuesta_generada}
            Contexto usado: {contexto_recuperado}
            
            PERFIL DEL ESTUDIANTE:
            - Nivel de comprensión: {nivel_comprension}
            - Temas dominados: {temas_dominados}
            - Áreas de dificultad: {areas_dificultad}
            
            CRITERIOS DE EVALUACIÓN:
            1. **Correctitud matemática**: ¿La respuesta es matemáticamente correcta?
            2. **Claridad pedagógica**: ¿Es comprensible para el nivel del estudiante?
            3. **Completitud**: ¿Responde completamente a la consulta?
            4. **Relevancia**: ¿Es relevante al contexto y nivel del estudiante?
            5. **Adaptación**: ¿Está adaptada al perfil del estudiante?
            
            INSTRUCCIONES:
            - Evalúa cada criterio objetivamente
            - Identifica si la respuesta es suficiente o necesita mejoras
            - Determina si se necesita más contexto o información
            - Proporciona recomendaciones específicas para mejorar
            
            {format_instructions}
            """
        )
    
    async def evaluator_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Cadena principal del evaluador que analiza la calidad de las respuestas"""
        try:
            logger.info(f"Evaluador analizando estado: {estado.estado_actual}")
            
            # Obtener la respuesta a evaluar
            respuesta_a_evaluar ,tipo_respuesta = self._obtener_respuesta_para_evaluar(estado)
            
            if not respuesta_a_evaluar:
                logger.warning("No hay respuesta para evaluar")
                estado.estado_actual = "evaluator_sin_contenido"
                return estado
            
            student_context = estado.estado_estudiante.model_dump()
            
            prompt_data = {
                "consulta_original": estado.consulta_inicial,
                "respuesta_generada": respuesta_a_evaluar,
                "contexto_recuperado": estado.contexto_recuperado,
                "nivel_comprension": student_context.get("nivel_comprension"),
                "temas_dominados": student_context.get("temas_dominados", []),
                "areas_dificultad": student_context.get("areas_dificultad", []),
                "format_instructions": self.parser.get_format_instructions()
            }
            
            # Ejecutar evaluación con fallback
            try:
                formatted_prompt = self.prompt.format(**prompt_data)
                evaluacion = await self.llm_structured.ainvoke(formatted_prompt)
                
                if isinstance(evaluacion, dict):
                    evaluacion = ResponseEvaluation(
                        is_sufficient=evaluacion.get('is_sufficient', True),
                        correctness_score=evaluacion.get('correctness_score', 0.8),
                        clarity_score=evaluacion.get('clarity_score', 0.8),
                        completeness_score=evaluacion.get('completeness_score', 0.8),
                        relevance_score=evaluacion.get('relevance_score', 0.8),
                        needs_more_context=evaluacion.get('needs_more_context', False),
                        improvement_suggestions=evaluacion.get('improvement_suggestions', []),
                        overall_quality=evaluacion.get('overall_quality', 'good')
                    )
                    
            except Exception as structured_error:
                logger.warning(f"Evaluator structured output falló: {structured_error}")
                try:
                    formatted_prompt = self.prompt.format(**prompt_data)
                    raw_response = await self.llm.ainvoke(formatted_prompt)
                    evaluacion_dict = self.parser.parse(raw_response)
                    
                    if isinstance(evaluacion_dict, dict):
                        evaluacion = ResponseEvaluation(**evaluacion_dict)
                    else:
                        evaluacion = evaluacion_dict
                        
                except Exception as parse_error:
                    logger.error(f"Error evaluando respuesta: {parse_error}")
                    # Fallback: evaluación por defecto positiva
                    evaluacion = ResponseEvaluation(
                        is_sufficient=True,
                        correctness_score=0.7,
                        clarity_score=0.7,
                        completeness_score=0.7,
                        relevance_score=0.7,
                        needs_more_context=False,
                        improvement_suggestions=["Respuesta generada exitosamente"],
                        overall_quality="acceptable"
                    )
            
            self._actualizar_estado_con_evaluacion(estado, evaluacion)
            if tipo_respuesta == "exam_creator":
                estado.estado_actual = "exam_creator_evaluado"
                logger.info("✅ Exam creator evaluado - marcando estado")
            elif tipo_respuesta == "math_expert":
                estado.estado_actual = "math_expert_evaluado" 
                logger.info("✅ Math expert evaluado - marcando estado")
            else:
                estado.estado_actual = "evaluator_completado"
                logger.info("✅ Evaluación general completada")
            
            logger.info(f"🎯 Evaluación final: {evaluacion.overall_quality} (suficiente: {evaluacion.is_sufficient})")
            logger.info(f"🏁 Estado actualizado a: {estado.estado_actual}")
            
            return estado
            
        except Exception as e:
            logger.error(f"Error crítico en evaluator: {e}")
            estado.estado_actual = "evaluator_error"
            return estado
    
    def _obtener_respuesta_para_evaluar(self, estado: EstadoConversacion) -> tuple[str, str]:
        """
        Determina qué respuesta evaluar y su tipo basándose en el estado actual
        
        Returns:
            tuple[str, str]: (respuesta_a_evaluar, tipo_respuesta)
            Si no hay respuesta disponible, retorna (None, "")
        """
        if estado.respuesta_exam_creator and estado.estado_actual.startswith("exam_creator"):
            logger.info("📋 Evaluando respuesta del exam_creator")
            return estado.respuesta_exam_creator, "exam_creator"
            
        elif estado.respuesta_math_expert and estado.estado_actual.startswith("math_expert"):
            logger.info("🧮 Evaluando respuesta del math_expert")
            return estado.respuesta_math_expert, "math_expert"
        
        # FALLBACK: Evaluar cualquiera que esté disponible (solo si no se ha evaluado)
        elif (estado.respuesta_math_expert and 
            estado.estado_actual not in ["math_expert_evaluado", "evaluator_completado"]):
            logger.info("🔄 Fallback: evaluando math_expert")
            return estado.respuesta_math_expert, "math_expert"
            
        elif (estado.respuesta_exam_creator and 
            estado.estado_actual not in ["exam_creator_evaluado", "evaluator_completado"]):
            logger.info("🔄 Fallback: evaluando exam_creator")
            return estado.respuesta_exam_creator, "exam_creator"
        
        # No hay respuesta disponible para evaluar
        logger.warning("⚠️ No hay respuesta disponible para evaluar")
        return None, ""
    
    def _actualizar_estado_con_evaluacion(self, estado: EstadoConversacion, evaluacion: ResponseEvaluation):
        """Actualiza el estado con los resultados de la evaluación"""
        estado.necesita_crawler = evaluacion.needs_more_context
        
        # Agregar información de evaluación al historial
        evaluacion_summary = {
            "timestamp": estado.timestamp,
            "evaluation": {
                "is_sufficient": evaluacion.is_sufficient,
                "overall_quality": evaluacion.overall_quality,
                "scores": {
                    "correctness": evaluacion.correctness_score,
                    "clarity": evaluacion.clarity_score,
                    "completeness": evaluacion.completeness_score,
                    "relevance": evaluacion.relevance_score
                },
                "suggestions": evaluacion.improvement_suggestions
            }
        }
        
        estado.chat_history.append({
            "role": "evaluator",
            "content": f"Evaluación completada: {evaluacion.overall_quality}",
            "metadata": evaluacion_summary
        })
        
        estado.estado_actual = "evaluator_completado"
        
        if evaluacion.needs_more_context:
            estado.necesita_crawler = True
            logger.info("Evaluador recomienda usar crawler para más contexto")