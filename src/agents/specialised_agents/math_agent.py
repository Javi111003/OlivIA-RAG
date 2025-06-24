from datetime import datetime
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from  agents.dto_s.agent_formated_responses import MathExpertResponse
from  agents.dto_s.agent_state import EstadoConversacion
from  generator.llm_provider import MistralLLMProvider
from agents.specialised_agents.knowledge_analyzer import KnowledgeAnalyzerAgent
import asyncio
import logging
from typing import Dict

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
            - Nivel de comprensión general: {nivel_comprension}
            - Puntuación general: {knowledge_areas[overall_score]}/10
            
            ÁREAS DE FORTALEZA (Score ≥ 7):
            {knowledge_areas[strong_areas]}
            
            ÁREAS DE DEBILIDAD (Score ≤ 4):
            {knowledge_areas[weak_areas]}
            
            CAMPOS LEGACY (para compatibilidad):
            - Temas dominados: {temas_dominados}
            - Áreas de dificultad: {areas_dificultad}
            - Errores históricos: {historial_errores}
            - Preferencias: {preferencias}
            
            CONTEXTO DE LA CONSULTA:
            - Consulta original: {consulta_inicial}
            - Contexto conversacional: {contexto_conversacional}
            - Contexto recuperado: {contexto_recuperado}
            - Estado BDI: {bdi_context}
            - Historial reciente: {chat_history}
            
            ANÁLISIS DE REFERENCIAS TEMPORALES:
            Revisa cuidadosamente si la consulta incluye referencias como:
            - "anteriormente", "antes", "previo", "mencionamos", "hablamos de"
            - "el teorema que", "la fórmula que", "el concepto que"
            - "continúa", "sigue", "siguiente", "más sobre"
            
            Si la consulta contiene referencias temporales:
            1. PRIORIDAD MÁXIMA: Busca en el historial conversacional (chat_history)
            2. Identifica qué teorema/concepto/ejercicio/examen se mencionó específicamente
            3. NO uses el contexto RAG recuperado cuando exista una referencia temporal a algo pues el usuario no sabe del contexto recuperado
            4. Si no encuentras la referencia en el historial, indica que no hay contexto previo
            
            INSTRUCCIONES PEDAGÓGICAS AVANZADAS:
            1. **Personalización por áreas**: Adapta la explicación considerando las puntuaciones específicas por área
            2. **Aprovecha fortalezas**: Conecta conceptos nuevos con áreas donde tiene buena puntuación
            3. **Refuerza debilidades**: Si la consulta toca áreas débiles, proporciona apoyo extra
            4. **Detecta progreso**: Identifica si la consulta muestra mejora en áreas previamente débiles
            5. **Contextualización**: Usa el contexto conversacional para referencias específicas
            
            INSTRUCCIONES IMPORTANTES:
            1. **PRIORIDAD**: Si la consulta referencia algo previo ("ejercicio 1", "tú examen","el teorema") o no aparece en la consulta el sujeto o cosa de la cual se habla, usa contexto conversacional únicamente.
            Si no hay referencia previa, usa el contexto general de conocimiento matemático.
            2. **AREAS ESPECÍFICAS**: Menciona qué área de conocimiento estás trabajando
            3. **PROGRESO**: Si detectas mejora, reconócelo explícitamente
            4. **CONEXIONES**: Conecta con áreas fuertes para facilitar comprensión
            
            EJEMPLOS DE RESPUESTA:
            - "Basándome en el examen que creé anteriormente..."
            - "Refiriéndome al ejercicio 1 del examen que generé..."
            - "En relación al teorema que discutimos..."
            - "Según el contexto de nuestra conversación previa..."
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
            
            knowledge_context = self._extract_knowledge_context(estado.estado_estudiante.math_knowledge)
            contexto_conversacional = self._extraer_contexto_conversacional(estado)

            prompt_data = {
                "nivel_comprension": student_context.get("nivel_comprension"),
                "temas_dominados": student_context.get("temas_dominados", []),
                "areas_dificultad": student_context.get("areas_dificultad", []),
                "historial_errores": student_context.get("historial_errores", []),
                "preferencias": student_context.get("preferencias_aprendizaje", {}),
                "knowledge_areas": knowledge_context,
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
            
            estado.respuesta_math_expert = respuesta.explanation
            estado.estado_actual = "math_expert_completado"
            
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
            #self.actualizar_perfil_estudiante(estado, respuesta)
            
            print("Actualizando perfil del estudiante...")
            await self._analyze_and_update_knowledge(estado, respuesta)
            
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
        
        for mensaje in estado.chat_history[-5:]:
            if mensaje.get("role") == "exam_creator":
                contexto_partes.append(f"EXAMEN CREADO PREVIAMENTE:\n{mensaje['content']}\n")
            elif mensaje.get("role") == "math_expert":
                contexto_partes.append(f"EXPLICACIÓN PREVIA:\n{mensaje['content'][:300]}...\n")
            elif mensaje.get("role") == "user":
                contexto_partes.append(f"CONSULTA DEL USUARIO:\n{mensaje['content'][:300]}...\n")
        
        if not contexto_partes:
            return "No hay contexto conversacional previo."
        
        return "\n".join(contexto_partes)
    
    def _extract_knowledge_context(self, math_knowledge) -> Dict:
        """Extrae contexto relevante de las áreas de conocimiento"""
        all_areas = math_knowledge.get_all_areas()
        
        # Áreas fuertes (score >= 7)
        strong_areas = [
            {"name": area.name, "score": area.score, "topics": area.topics_mastered}
            for area in all_areas.values() if area.score >= 7
        ]
        
        # Áreas débiles (score <= 4)
        weak_areas = [
            {"name": area.name, "score": area.score, "struggles": area.topics_struggling}
            for area in all_areas.values() if area.score <= 4
        ]
        
        # Puntuación general
        overall_score = math_knowledge.get_overall_score()
        
        return {
            "overall_score": round(overall_score, 1),
            "strong_areas": strong_areas,
            "weak_areas": weak_areas,
            "relevant_areas": [area.name for area in all_areas.values()]
        }
    
    async def _analyze_and_update_knowledge(self, estado: EstadoConversacion, respuesta: MathExpertResponse):
        """Analiza la interacción y actualiza el conocimiento del estudiante"""
        try:
            # Crear analizador (lazy import para evitar ciclos)            
            analyzer = KnowledgeAnalyzerAgent(self.llm)
            
            analysis = await analyzer.analyze_knowledge_from_interaction(estado)
            print(f"🔍🔍🔍Análisis de conocimiento: {analysis}")
            if analysis:
                estado = analyzer.update_student_knowledge(estado, analysis)
                logger.info("✅ Conocimiento del estudiante actualizado automáticamente")
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis automático de conocimiento: {e}")