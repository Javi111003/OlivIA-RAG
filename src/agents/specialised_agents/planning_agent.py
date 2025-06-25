from datetime import datetime
from typing import Any, Dict, List
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from  agents.dto_s.agent_formated_responses import PlanningResponse , PlanBlock
from  agents.dto_s.agent_state import EstadoConversacion 
from  generator.llm_provider import MistralLLMProvider
from  planner.entities import *
from  planner.genetic_algorithm import *
from  planner.evaluation import *
import json
import random

logger = logging.getLogger(__name__)

class PlanningAgent:
    """Agente planificador para generar rutas de estudio"""
    
    def __init__(self, llm: MistralLLMProvider):
        self.llm = llm
        self.llm_structured = llm.with_structured_output(PlanningResponse)
        self.parser = JsonOutputParser(pydantic_object=PlanningResponse)
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un experto creador de planes de estudio. Tu tarea es recibir un plan de estudio generado por otro experto y mostrarlo al usuario de forma clara y organizada.

            PLAN PARA EL ESTUDIANTE: {topics_data}
            
            PUNTUACIÓN DEL PLAN: {score}
            
            INSTRUCCIONES:
            - Recibirás el plan en el siguiente formato: Nombre del tema : Tiempo Dedicado (separados por coma por cada tema)
            - Después del plan recibirás un valor numérico representando su calidad.
            - El orden de estudio de las asignaturas corresponde con el orden en el que las recibes.

            INSTRUCCIONES IMPORTANTES:
                1. Responde ÚNICAMENTE con JSON válido
                2. NO uses markdown, NO agregues texto extra
                3. Usa comillas dobles para strings
                4. Evita saltos de línea dentro de strings
                5. Mantén las preguntas concisas

            FORMATO DE RESPUESTA:
            - Devuelve el plan estructurado en el siguiente formato JSON

            {{
                "plan": [
                    {{
                        "topic": "Nombre del tema",
                        "topic_description": "Descripción explicativa del tema",
                        "time_allocated": tiempo_en_horas
                    }},
                    ...
                ],
                "score": puntuacion_del_plan
            }}

            IMPORTANTE:
            - El campo "plan" debe ser una lista de objetos, no un diccionario.
            - Cada objeto de la lista plan debe tener : topic (string), topic_description (string), time_allocated (float).
            - Responde ÚNICAMENTE con JSON válido, sin markdown, sin comillas adicionales ni texto extra.
            - No incluyas comentarios ni explicaciones fuera del JSON.

            NO uses saltos de línea ni indentación, responde TODO en una sola línea de JSON válido.

            {format_instructions}
            """
        )
    
    async def plannig_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        # Obtener datos del estudiante
        student_context = estado.estado_estudiante
        topics = student_context.math_knowledge.get_all_areas().values()   
        topics_models = {}
        scores = {}
        
        for topic in topics:
            t = Topic(
                name= topic.name,
                exam_weight= topic.weight,
                base_difficulty= topic.difficulty
            )
            topics_models[t.name] = t
            scores[t.name] = topic.score
        
        student = Student(
            topic_mastery= scores,
            target_score= 100
        )

        initial_population = generate_population(random.randint(50,100), topics_models, student, 40, 1, len(topics))
        _ , best_plan = evolve_population(initial_population, evaluate_plan, topics=topics_models, student=student, num_generations=5)

        topics_data = ""

        for topic_block in best_plan.blocks:
            topics_data += f"{topic_block.topic.name} : {topic_block.time_allocated} , " 

        prompt_data = {
            "topics_data" : topics_data ,
            "score" :  evaluate_plan(best_plan, student, topics_models),
            "format_instructions": self.parser.get_format_instructions()
        }
        respuesta_raw = None
        try:
            formatted_prompt = self.prompt.format(**prompt_data) 
            respuesta_raw = await self.llm_structured.ainvoke(formatted_prompt)
            print(f"✅ Structured output exitoso: {type(respuesta_raw)}")
        except Exception as structured_error:
            logger.warning(f"⚠️ Math expert structured output falló: {structured_error}")
            print("Respuesta cruda del modelo (structured):", respuesta_raw)
            try:
                formatted_prompt = self.prompt.format(**prompt_data)
                respuesta_raw = await self.llm.ainvoke(formatted_prompt)
                print(f"📝 Raw response obtenida: {type(respuesta_raw)}")
                # Intentar parsing manual
                if isinstance(respuesta_raw, str):
                    print("Respuesta cruda del modelo (raw):", respuesta_raw)
                    try:
                        respuesta_raw = json.loads(str(respuesta_raw))
                        logger.info("✅ JSON parsing manual exitoso")
                    except json.JSONDecodeError:
                        logger.warning("⚠️ No se pudo parsear JSON manualmente")
            except Exception as raw_error:
                logger.error(f"❌ Raw response también falló: {raw_error}")
                respuesta_raw = {}

        # Convertir a formato estándar
        planning = self.ensure_planning_response(respuesta_raw)

        # Actualizar estado
        estado.respuesta_planning = self.format_planning_output(planning)
        estado.estado_actual = "planning_agent_completado"

        # Agregar al historial
        estado.chat_history.append({
            "role": "planning_agent",
            "content": estado.respuesta_planning,
            "metadata": {
                "plan": planning.plan,
                "score": planning.score,
                "timestamp": datetime.now().isoformat(),
                "personalization_applied": True
            }
        })

        logger.info(f"✅ PlanningAgent completado (score: {planning.score})")
        return estado
    
    def ensure_planning_response(self, respuesta):
        """Convierte la respuesta al formato deseado (lista de objetos PlanBlock)"""
        try:
            if isinstance(respuesta, PlanningResponse):
                return respuesta

            # Si es dict, intenta extraer la lista
            if isinstance(respuesta, dict):
                plan = respuesta.get("plan", [])
                # Si por error viene como dict de temas, conviértelo a lista de PlanBlock
                if isinstance(plan, dict):
                    plan = [
                        {
                            "topic": topic,
                            "topic_description": data.get("topic_description", ""),
                            "time_allocated": data.get("time_allocated", 0.0)
                        }
                        for topic, data in plan.items()
                    ]
                # Si ya es lista, intenta convertir cada item a PlanBlock
                plan_blocks = [PlanBlock(**item) if not isinstance(item, PlanBlock) else item for item in plan]
                return PlanningResponse(
                    plan=plan_blocks,
                    score=respuesta.get("score", 0.0)
                )

            # Si es string, intenta parsear JSON y repetir lógica
            if isinstance(respuesta, str):
                try:
                    json_data = json.loads(respuesta)
                    plan = json_data.get("plan", [])
                    if isinstance(plan, dict):
                        plan = [
                            {
                                "topic": topic,
                                "topic_description": data.get("topic_description", ""),
                                "time_allocated": data.get("time_allocated", 0.0)
                            }
                            for topic, data in plan.items()
                        ]
                    plan_blocks = [PlanBlock(**item) if not isinstance(item, PlanBlock) else item for item in plan]
                    return PlanningResponse(
                        plan=plan_blocks,
                        score=json_data.get("score", 0.0)
                    )
                except json.JSONDecodeError:
                    pass

            logger.warning(f"No se pudo convertir respuesta de tipo {type(respuesta)}, usando fallback")
            return PlanningResponse(
                plan=[],
                score=-1
            )
        except Exception as e:
            logger.error(f"Error convirtiendo respuesta: {e}")
            return PlanningResponse(
                plan=[],
                score=-1
            )
        
    def format_planning_output(self, planning: PlanningResponse) -> str:
        """Formatea la salida del plan de estudio para mostrar al usuario"""
        output = "# Plan de Estudio Personalizado\n\n"
        output += f"**Puntuación del plan:** {planning.score}\n\n"
        output += "## Temas y tiempos sugeridos:\n\n"
        for i, block in enumerate(planning.plan, 1):
            output += f"**{i}.** **{block.topic}** — **{block.time_allocated}** horas\n\n"
            if block.topic_description:
                output += f"    - {block.topic_description}\n\n"
        output += "\n---\n\n*Plan generado automáticamente por OlivIA*"
        return output