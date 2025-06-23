import logging
from datetime import datetime
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.dto_s.agent_state import EstadoConversacion 
from agents.dto_s.agent_formated_responses import ExamCreatorResponse
from generator.llm_provider import MistralLLMProvider

logger = logging.getLogger(__name__)

class ExamCreatorAgent:
    """Agente creador de exámenes que genera evaluaciones personalizadas"""
    
    def __init__(self, llm: MistralLLMProvider):
        self.llm = llm
        self.llm_structured = llm.with_structured_output(ExamCreatorResponse)
        self.parser = JsonOutputParser(pydantic_object=ExamCreatorResponse)
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un experto en la creación de exámenes matemáticos personalizados para estudiantes universitarios.
            
            SOLICITUD DEL USUARIO:
            Consulta original: {consulta_inicial}
            Contexto recuperado: {contexto_recuperado}
            
            PERFIL DEL ESTUDIANTE:
            - Nivel de comprensión: {nivel_comprension}
            - Temas dominados: {temas_dominados}
            - Áreas de dificultad: {areas_dificultad}
            - Preferencias de aprendizaje: {preferencias}
            
            INSTRUCCIONES PARA CREAR EL EXAMEN:
            1. **Personalización**: Adapta la dificultad al nivel del estudiante
            2. **Diversidad**: Incluye diferentes tipos de preguntas (conceptuales, procedimentales, aplicación)
            3. **Progresión**: Ordena las preguntas de menor a mayor dificultad
            4. **Cobertura**: Asegúrate de cubrir los temas principales solicitados
            5. **Claridad**: Preguntas claras y sin ambigüedades
            6. **Tiempo realista**: Estima un tiempo apropiado para completar el examen
            
            TIPOS DE PREGUNTAS A INCLUIR:
            - Definiciones y conceptos fundamentales
            - Procedimientos y cálculos
            - Problemas de aplicación práctica
            - Análisis y demostración (si es nivel avanzado)
            
            INSTRUCCIONES IMPORTANTES:
            1. Responde ÚNICAMENTE con JSON válido
            2. NO uses markdown, NO agregues texto extra
            3. Usa comillas dobles para strings
            4. Evita saltos de línea dentro de strings
            5. Mantén las preguntas concisas
            
            Formato JSON requerido:
            {{
                "exam_title": "Título del examen aquí",
                "questions": ["Pregunta 1", "Pregunta 2", "Pregunta 3"],
                "difficulty_level": "básico",
                "estimated_time": 60,
                "topic_coverage": ["tema1", "tema2"]
            }}
            
            SOLO JSON, nada más.
            """
        )
    
    def extract_exam_from_response(self, response_text: str) -> dict:
        """Extrae JSON del examen de la respuesta del LLM"""
        try:
            logger.info(f"🔧 Extrayendo examen de respuesta...")
            
            if isinstance(response_text, dict):
                return response_text
            
            response_str = str(response_text)
            cleaned = response_str.strip()
            
            # Remover markdown si existe
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Buscar JSON con regex
            import re
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    import json
                    parsed_json = json.loads(json_str)
                    logger.info(f"✅ Examen extraído exitosamente")
                    return parsed_json
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Error parseando JSON del examen: {e}")
                    return {}
            
            try:
                import json
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"❌ No se pudo extraer JSON válido: {e}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error en extracción manual: {e}")
            return {}
    
    def create_fallback_exam(self, consulta: str, nivel: str) -> ExamCreatorResponse:
        """Crea un examen de fallback cuando el parsing falla"""
        try:
            # Preguntas básicas por defecto
            preguntas_basicas = [
                f"Define los conceptos fundamentales relacionados con: {consulta}",
                f"Explica paso a paso un procedimiento básico sobre: {consulta}",
                f"Resuelve un problema simple aplicando los conceptos de: {consulta}"
            ]
            
            preguntas_intermedias = [
                f"Analiza y compara diferentes enfoques para resolver problemas de: {consulta}",
                f"Demuestra la validez de las fórmulas principales en: {consulta}",
                f"Aplica los conceptos de {consulta} a un problema del mundo real"
            ]
            
            preguntas_avanzadas = [
                f"Desarrolla una demostración formal relacionada con: {consulta}",
                f"Analiza críticamente las limitaciones y extensiones de: {consulta}",
                f"Propone una generalización o variación de los conceptos de: {consulta}"
            ]
            
            if nivel == "principiante":
                preguntas = preguntas_basicas
                tiempo = 45
            elif nivel == "intermedio":
                preguntas = preguntas_basicas[:2] + preguntas_intermedias[:2]
                tiempo = 75
            else:  # avanzado
                preguntas = preguntas_intermedias + preguntas_avanzadas[:2]
                tiempo = 90
            
            return ExamCreatorResponse(
                exam_title=f"Examen sobre {consulta}",
                questions=preguntas,
                difficulty_level={"principiante": "básico", "intermedio": "intermedio", "avanzado": "avanzado"}[nivel],
                estimated_time=tiempo,
                topic_coverage=[consulta, "Conceptos fundamentales"]
            )
            
        except Exception as e:
            logger.error(f"❌ Error creando examen de fallback: {e}")
            return ExamCreatorResponse(
                exam_title="Examen Matemático",
                questions=["Pregunta 1: Define los conceptos básicos", "Pregunta 2: Resuelve un ejercicio simple"],
                difficulty_level="básico",
                estimated_time=30,
                topic_coverage=["Matemáticas básicas"]
            )
    
    def ensure_exam_creator_response(self, respuesta, consulta: str, nivel: str) -> ExamCreatorResponse:
        """Convierte cualquier formato de respuesta a ExamCreatorResponse"""
        try:
            # Si ya es ExamCreatorResponse
            if isinstance(respuesta, ExamCreatorResponse):
                logger.info("✅ Respuesta ya es ExamCreatorResponse")
                return respuesta
            
            if isinstance(respuesta, dict) and 'exam_title' in respuesta:
                logger.info("🔄 Convirtiendo dict a ExamCreatorResponse")
                return ExamCreatorResponse(
                    exam_title=respuesta.get('exam_title', f'Examen sobre {consulta}'),
                    questions=respuesta.get('questions', []),
                    difficulty_level=respuesta.get('difficulty_level', 'básico'),
                    estimated_time=respuesta.get('estimated_time', 60),
                    topic_coverage=respuesta.get('topic_coverage', [])
                )
            
            if isinstance(respuesta, str):
                logger.info(f"🔧 Intentando parsear string de examen...")
                json_data = self.extract_exam_from_response(respuesta)
                if json_data and 'exam_title' in json_data:
                    return ExamCreatorResponse(
                        exam_title=json_data.get('exam_title', f'Examen sobre {consulta}'),
                        questions=json_data.get('questions', []),
                        difficulty_level=json_data.get('difficulty_level', 'básico'),
                        estimated_time=json_data.get('estimated_time', 60),
                        topic_coverage=json_data.get('topic_coverage', [])
                    )
            
            # Fallback
            logger.warning(f"⚠️ No se pudo convertir respuesta, usando fallback")
            return self.create_fallback_exam(consulta, nivel)
            
        except Exception as e:
            logger.error(f"❌ Error convirtiendo respuesta de examen: {e}")
            return self.create_fallback_exam(consulta, nivel)
    
    async def exam_creator_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Cadena principal del creador de exámenes"""
        logger.info(f"ExamCreator procesando: {estado.consulta_inicial}")
        
        try:
            student_context = estado.estado_estudiante.model_dump()
            
            prompt_data = {
                "consulta_inicial": estado.consulta_inicial,
                "contexto_recuperado": str(estado.contexto_recuperado),
                "nivel_comprension": student_context.get("nivel_comprension", "principiante"),
                "temas_dominados": student_context.get("temas_dominados", []),
                "areas_dificultad": student_context.get("areas_dificultad", []),
                "preferencias": student_context.get("preferencias_aprendizaje", {})
            }
            
            respuesta_raw = None
            formatted_prompt = self.prompt.format(**prompt_data)
            
            # Intentar structured output primero
            try:
                logger.info("🔧 ExamCreator: Intentando structured output...")
                respuesta_raw = await self.llm_structured.ainvoke(formatted_prompt)
                logger.info(f"✅ Structured output exitoso: {type(respuesta_raw)}")
                
            except Exception as structured_error:
                logger.warning(f"⚠️ Structured output falló: {structured_error}")
                
                # Fallback a raw response
                try:
                    logger.info("🔄 Intentando raw response...")
                    respuesta_raw = await self.llm.ainvoke(formatted_prompt)
                    logger.info(f"📝 Raw response obtenida: {type(respuesta_raw)}")
                except Exception as raw_error:
                    logger.error(f"❌ Raw response también falló: {raw_error}")
                    respuesta_raw = {}
            
            # Convertir a formato estándar
            examen = self.ensure_exam_creator_response(
                respuesta_raw, 
                estado.consulta_inicial,
                student_context.get("nivel_comprension", "principiante")
            )
            
            # Actualizar estado
            estado.respuesta_exam_creator = self._format_exam_output(examen)
            estado.estado_actual = "exam_creator_completado"
            
            # Agregar al historial
            estado.chat_history.append({
                "role": "exam_creator",
                "content": estado.respuesta_exam_creator,
                "metadata": {
                    "exam_title": examen.exam_title,
                    "num_questions": len(examen.questions),
                    "difficulty": examen.difficulty_level,
                    "estimated_time": examen.estimated_time,
                    "topics": examen.topic_coverage,
                    "timestamp": datetime.now().isoformat(),
                    "personalization_applied": True
                }
            })
            
            logger.info(f"✅ ExamCreator completado: {examen.exam_title} ({len(examen.questions)} preguntas)")
            return estado
            
        except Exception as e:
            logger.error(f"💥 Error crítico en exam creator: {e}")
            import traceback
            traceback.print_exc()
            
            # Respuesta de emergencia
            estado.respuesta_exam_creator = f"Examen sobre: {estado.consulta_inicial}\n\n1. Explica los conceptos fundamentales\n2. Resuelve un ejercicio básico\n3. Aplica los conocimientos a un problema práctico"
            estado.estado_actual = "exam_creator_completado"
            
            estado.chat_history.append({
                "role": "exam_creator",
                "content": estado.respuesta_exam_creator,
                "metadata": {
                    "exam_title": f"Examen sobre {estado.consulta_inicial}",
                    "num_questions": 3,
                    "difficulty": "básico",
                    "estimated_time": 45,
                    "topics": [estado.consulta_inicial],
                    "timestamp": datetime.now().isoformat(),
                    "error_recovery": True
                }
            })
            
            return estado
    
    def _format_exam_output(self, examen: ExamCreatorResponse) -> str:
        """Formatea la salida del examen para mostrar al usuario"""
        output = f"# {examen.exam_title}\n\n"
        output += f"**Nivel de dificultad:** {examen.difficulty_level.title()}\n"
        output += f"**Tiempo estimado:** {examen.estimated_time} minutos\n"
        output += f"**Temas cubiertos:** {', '.join(examen.topic_coverage)}\n\n"
        output += "## Preguntas:\n\n"
        
        for i, pregunta in enumerate(examen.questions, 1):
            output += f"**{i}.** {pregunta}\n\n"
        
        output += "---\n\n*Examen generado automáticamente por OlivIA*"
        
        return output