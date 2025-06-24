from typing import Any, Dict, List
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from  agents.dto_s.agent_formated_responses import PlanningResponse
from  agents.dto_s.agent_state import EstadoConversacion 
from  generator.llm_provider import MistralLLMProvider
from  src.planner.entities import *
from  src.planner.genetic_algorithm import *
from  src.planner.evaluation import *
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
            Estás trabajado con un experto creardor de planes de estudio. Tu trabajo es recibir un plan de estudio generado por
            dicho experto y mostrarlo al usuario de una forma elegante y visible.

            PLAN PARA EL ESTUDIANTE : {topics_data}
            
            PUNTUACIÓN DEL PLAN : {score}
            
            INSTRUCCIONES:
             - Recibirás el plan en el siguiente formato: Nombre del tema : Tiempo Dedicado (separados por coma por cada tema)
             - Después del plan recibirás un valor númerico representando su calidad.
             - El orden de estudio de las asignaturas corresponde con el orden en el que las recibes.

            FORMATO DE RESPUESTA:
             - Estructuración organizada del plan recibido
            
            IMPORTANTE: Responde ÚNICAMENTE con JSON válido, sin markdown, sin comillas adicionales.
            
            {format_instructions}
            """
        )
    
    async def plannig_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        # Obtener datos del estudiante
        student_context = estado.estado_estudiante
        topics = student_context.math_knowledge.get_all_areas().values()   
        topics_models = []
        scores = {}
        
        for topic in topics:
            t = Topic(
                name= topic.name,
                exam_weight= topic.weight,
                base_difficulty= topic.difficulty
            )
            topics_models.append(t)
            scores[t.name] = topic.score
        
        student = Student(
            topic_mastery= scores,
            target_score= 100
        )

        initial_population = generate_population(random.randint(50,100), topics_models, student, 40, 1, len(topics))
        _ , best_plan = evolve_population(initial_population, evaluate_plan, 5)

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
    
    def ensure_planning_response(self, respuesta):
        """Cnvierte la respuesta al formato deseado"""
        try:
            if isinstance(respuesta, PlanningResponse):
                return respuesta
            
            if isinstance(respuesta, dict):
                return PlanningResponse(
                    plan= respuesta.get("plan", {}),
                    score= respuesta.get("score", 0.0)
                )
            
            if isinstance(respuesta, str):
                try:
                    json_data = json.loads(respuesta)
                    return PlanningResponse(
                        plan= json_data.get("plan", {}),
                        score= json_data.get("score", 0.0)
                    )
                except json.JSONDecodeError:
                    pass
            logger.warning(f"No se pudo convertir respuesta de tipo {type(respuesta)}, usando fallback")
            return PlanningResponse(
                plan= {},
                score= -1
            )
        except Exception as e:
            logger.error(f"Error convirtiendo respuesta: {e}")
            return PlanningResponse(
                plan= {},
                score= -1
            )