from datetime import datetime
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from  agents.dto_s.agent_formated_responses import MathExpertResponse
from  agents.dto_s.agent_state import EstadoConversacion
from  generator.llm_provider import MistralLLMProvider
import asyncio
import logging

logger = logging.getLogger(__name__)

class MathExpert():
    
    def __init__(self, llm: MistralLLMProvider):
        """Factory function para crear agente matemático avanzado"""
        self.llm = llm
        self.llm_structured = llm.with_structured_output(MathExpertResponse)
        self.parser = JsonOutputParser(pydantic_object=MathExpertResponse)

        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un experto en matemáticas con enfoque pedagógico personalizado.
            
            PERFIL DEL ESTUDIANTE:
            - Nivel de comprensión: {nivel_comprension}
            - Temas ya dominados: {temas_dominados}
            - Áreas de dificultad conocidas: {areas_dificultad}
            - Errores históricos: {historial_errores}
            - Preferencias de aprendizaje: {preferencias}
            
            CONTEXTO DE LA CONSULTA:
            - Consulta original: {consulta_inicial}
            - Contexto recuperado: {contexto_recuperado}
            - Estado BDI actual: {bdi_context}
            - Historial de conversación: {chat_history}
            
            INSTRUCCIONES PEDAGÓGICAS:
            1. Adapta la explicación al nivel del estudiante
            2. Conecta con conocimientos previos (temas dominados)
            3. Anticipa confusiones basándote en errores históricos
            4. Usa el estilo de aprendizaje preferido
            5. Proporciona fórmulas relevantes y conceptos relacionados
            6. Incluye verificaciones de comprensión si es apropiado
            
            INSTRUCCIONES IMPORTANTES:
            1. **PRIORIDAD**: Si la consulta hace referencia a algo mencionado en la conversación previa (como "ejercicio 1", "pregunta 1", "tu examen"), usa ÚNICAMENTE el contexto conversacional, NO el contexto recuperado.
            
            2. Si la consulta menciona "ejercicio", "pregunta", "problema" + número, busca específicamente esa referencia en el contexto conversacional.
            
            3. Si no encuentras la referencia específica en la conversación previa, entonces puedes usar el contexto recuperado.
            
            4. Siempre menciona de dónde viene la información que estás usando.
            
            EJEMPLOS DE RESPUESTA:
            - "Basándome en el examen que creé anteriormente..."
            - "Refiriéndome al ejercicio 1 del examen que generé..."
            - "Como no encuentro referencia previa específica, te ayudo con información general..."
            
            
            FORMATO DE RESPUESTA:
            - explanation: Explicación detallada y personalizada
            - formulas: Lista de fórmulas relevantes (LaTeX si aplica)
            - difficulty_level: Evalúa la dificultad para este estudiante específico
            - related_concepts: Conceptos que conectan con conocimiento previo
            
            IMPORTANTE: Responde ÚNICAMENTE con JSON válido, sin markdown, sin comillas adicionales.

            {format_instructions}
            """
        )

    async def math_expert_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Cadena principal del experto matemático"""
        print(f"Math Expert procesando: {estado.consulta_inicial}")
        try:
            # Preparar contexto personalizado
            student_context = estado.estado_estudiante.model_dump()
            bdi_context = estado.bdi_state.model_dump() if estado.bdi_state else {}
            
            contexto_conversacional = self._extraer_contexto_conversacional(estado)

            prompt_data = {
                "nivel_comprension": student_context.get("nivel_comprension"),
                "temas_dominados": student_context.get("temas_dominados", []),
                "areas_dificultad": student_context.get("areas_dificultad", []),
                "historial_errores": student_context.get("historial_errores", []),
                "preferencias": student_context.get("preferencias_aprendizaje", {}),
                "consulta_inicial": estado.consulta_inicial,
                "contexto_conversacional": contexto_conversacional,
                "contexto_recuperado": estado.contexto_recuperado,
                "bdi_context": bdi_context,
                "chat_history": estado.chat_history[-5:],  # Últimas 5 interacciones
                "format_instructions": self.parser.get_format_instructions()
            }
            
            respuesta_raw = None
            # Ejecutar con fallback
            try:
                formatted_prompt = self.prompt.format(**prompt_data)
                respuesta_raw = await self.llm_structured.ainvoke(formatted_prompt)
                logger.info(f"✅ Structured output exitoso: {type(respuesta_raw)}")
                print(repr(respuesta_raw))  # Log completo de la respuesta
                print(f"Respuesta del experto matemático: {respuesta_raw.explanation[:100]}...")  # Log parcial
            except Exception as structured_error:
                logger.warning(f"⚠️ Math expert structured output falló: {structured_error}")
                try:
                    formatted_prompt = self.prompt.format(**prompt_data)
                    respuesta_raw = await self.llm.ainvoke(formatted_prompt)
                    logger.info(f"📝 Raw response obtenida: {type(respuesta_raw)}")
                    
                    # Intentar parsing manual
                    if isinstance(respuesta_raw, str):
                        try:
                            respuesta_raw = json.loads(str(respuesta_raw))
                            logger.info("✅ JSON parsing manual exitoso")
                        except json.JSONDecodeError:
                            logger.warning("⚠️ No se pudo parsear JSON manualmente")
                    
                except Exception as raw_error:
                    logger.error(f"❌ Raw response también falló: {raw_error}")
                    respuesta_raw = {}
            
            respuesta = self.ensure_math_expert_response(respuesta_raw, estado.consulta_inicial)
            
            logger.info(f"🎯 Respuesta final normalizada: {type(respuesta)}")
            logger.info(f"📝 Explicación: {respuesta.explanation[:100]}...")
            
            # Actualizar estado
            estado.respuesta_math_expert = respuesta.explanation
            estado.estado_actual = "math_expert_completado"
            
            # Actualizar historial con metadatos ricos
            estado.chat_history.append({
                "role": "math_expert",
                "content": respuesta.explanation,
                "metadata": {
                    "formulas": respuesta.formulas,
                    "difficulty_assessed": respuesta.difficulty_level,
                    "related_concepts": respuesta.related_concepts,
                    "timestamp": datetime.now().isoformat(),
                    "personalization_applied": True
                }
            })
            
            # Actualizar perfil del estudiante basado en la interacción
            self.actualizar_perfil_estudiante(estado, respuesta)
            
            logger.info(f"Math expert completó respuesta (dificultad: {respuesta.difficulty_level})")
            return estado
            
        except Exception as e:
            logger.error(f"Error en math expert: {e}")
            estado.respuesta_math_expert = "Lo siento, ocurrió un error al procesar tu consulta matemática."
            estado.estado_actual = "math_expert_error"
            return estado
    

    def actualizar_perfil_estudiante(self, estado: EstadoConversacion, respuesta: MathExpertResponse) -> None:
        """Actualiza el perfil del estudiante basado en la interacción"""
        # Actualizar temas dominados si la dificultad es apropiada
        if respuesta.difficulty_level == "básico":
            for concepto in respuesta.related_concepts:
                if concepto not in estado.estado_estudiante.temas_dominados:
                    estado.estado_estudiante.temas_dominados.append(concepto)
        
        # Ajustar nivel de comprensión si es necesario
        if respuesta.difficulty_level == "avanzado" and estado.estado_estudiante.nivel_comprension == "principiante":
            estado.estado_estudiante.nivel_comprension = "intermedio"
    
    def ensure_math_expert_response(self, respuesta, consulta: str) -> MathExpertResponse:
        """Convierte cualquier formato de respuesta a MathExpertResponse"""
        try:
            # Si ya es MathExpertResponse, devolverlo tal cual
            if isinstance(respuesta, MathExpertResponse):
                return respuesta
            
            # Si es dict, convertir a MathExpertResponse
            if isinstance(respuesta, dict):
                return MathExpertResponse(
                    explanation=respuesta.get('explanation', f'Explicación sobre: {consulta}'),
                    formulas=respuesta.get('formulas', []),
                    difficulty_level=respuesta.get('difficulty_level', 'básico'),
                    related_concepts=respuesta.get('related_concepts', [])
                )
            
            # Si es string (JSON), parsearlo primero
            if isinstance(respuesta, str):
                try:
                    json_data = json.loads(respuesta)
                    return MathExpertResponse(
                        explanation=json_data.get('explanation', f'Explicación sobre: {consulta}'),
                        formulas=json_data.get('formulas', []),
                        difficulty_level=json_data.get('difficulty_level', 'básico'),
                        related_concepts=json_data.get('related_concepts', [])
                    )
                except json.JSONDecodeError:
                    pass
            
            # Fallback si no se puede convertir
            logger.warning(f"No se pudo convertir respuesta de tipo {type(respuesta)}, usando fallback")
            return MathExpertResponse(
                explanation=f"Respuesta sobre: {consulta}. {str(respuesta)[:200]}",
                formulas=[],
                difficulty_level='básico',
                related_concepts=[]
            )
            
        except Exception as e:
            logger.error(f"Error convirtiendo respuesta: {e}")
            return MathExpertResponse(
                explanation=f"Error procesando consulta sobre: {consulta}",
                formulas=[],
                difficulty_level='básico',
                related_concepts=[]
            )
    def _extraer_contexto_conversacional(self, estado: EstadoConversacion) -> str:
        """Extrae contexto relevante de la conversación previa"""
        contexto_partes = []
        
        # Buscar respuestas previas de agentes
        for mensaje in estado.chat_history[-5:]:
            if mensaje.get("role") == "exam_creator":
                contexto_partes.append(f"EXAMEN CREADO PREVIAMENTE:\n{mensaje['content']}\n")
            elif mensaje.get("role") == "math_expert":
                contexto_partes.append(f"EXPLICACIÓN PREVIA:\n{mensaje['content'][:300]}...\n")
        
        if not contexto_partes:
            return "No hay contexto conversacional previo."
        
        return "\n".join(contexto_partes)