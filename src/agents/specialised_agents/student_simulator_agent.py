from typing import Any, Dict, List, Tuple
import logging
import random
import numpy as np
from datetime import datetime
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.dto_s.agent_state import EstadoConversacion 
from agents.dto_s.agent_formated_responses import StudentSimulationResponse
from generator.llm_provider import MistralLLMProvider

logger = logging.getLogger(__name__)

class StudentSimulatorAgent:
    """Agente simulador de estudiante para experimentación pedagógica y optimización de planes de estudio"""
    
    def __init__(self, llm: MistralLLMProvider):
        self.llm = llm
        self.llm_structured = llm.with_structured_output(StudentSimulationResponse)
        self.parser = JsonOutputParser(pydantic_object=StudentSimulationResponse)
        
        # Métricas base para experimentación
        self.metrics_weights = {
            "knowledge_acquisition": 0.25,
            "retention_rate": 0.20,
            "engagement_level": 0.15,
            "error_reduction": 0.15,
            "confidence_growth": 0.10,
            "time_efficiency": 0.10,
            "transfer_learning": 0.05
        }
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un simulador de estudiante avanzado que modela el comportamiento de aprendizaje realista.
            
            CONTEXTO DE SIMULACIÓN:
            Consulta original: {consulta_inicial}
            Material presentado: {material_presentado}
            Tipo de contenido: {tipo_contenido}
            
            PERFIL DEL ESTUDIANTE A SIMULAR:
            - Nivel actual: {nivel_comprension}
            - Conocimientos previos: {temas_dominados}
            - Dificultades conocidas: {areas_dificultad}
            - Errores recurrentes: {historial_errores}
            - Estilo de aprendizaje: {estilo_aprendizaje}
            
            PARÁMETROS DE SIMULACIÓN:
            - Fatiga cognitiva: {fatiga_cognitiva}
            - Motivación actual: {motivacion}
            - Tiempo de estudio: {tiempo_estudio}
            - Contexto de aprendizaje: {contexto_aprendizaje}
            
            INSTRUCCIONES DE SIMULACIÓN:
            1. **Simula una respuesta realista** del estudiante basada en su perfil
            2. **Incluye errores típicos** que cometería alguien de su nivel
            3. **Refleja el proceso de razonamiento** paso a paso
            4. **Identifica puntos de confusión** específicos
            5. **Estima métricas de aprendizaje** realistas
            
            MÉTRICAS A CALCULAR:
            - Comprensión lograda (0-100%)
            - Retención estimada (0-100%)
            - Nivel de engagement (0-100%)
            - Reducción de errores (0-100%)
            - Crecimiento de confianza (0-100%)
            - Eficiencia temporal (0-100%)
            - Transferencia de conocimiento (0-100%)
            
            Responde ÚNICAMENTE con JSON válido:
            {{
                "student_answer": "Respuesta simulada del estudiante",
                "confidence_level": "baja/media/alta",
                "common_mistakes": ["error1", "error2"],
                "reasoning_process": "Proceso de razonamiento detallado",
                "areas_of_confusion": ["área1", "área2"],
                "learning_metrics": {{
                    "knowledge_acquisition": 75,
                    "retention_rate": 60,
                    "engagement_level": 80,
                    "error_reduction": 45,
                    "confidence_growth": 30,
                    "time_efficiency": 70,
                    "transfer_learning": 50
                }},
                "cognitive_load": "bajo/medio/alto",
                "motivation_change": -5,
                "predicted_performance": 72
            }}
            """
        )
    
    def generate_student_profile_variations(self, base_profile: Dict) -> List[Dict]:
        """Genera variaciones del perfil del estudiante para experimentación"""
        variations = []
        
        # Variación 1: Estudiante con más dificultades
        struggling_profile = base_profile.copy()
        struggling_profile.update({
            "nivel_comprension": "principiante",
            "fatiga_cognitiva": 0.8,
            "motivacion": 0.4,
            "areas_dificultad": base_profile.get("areas_dificultad", []) + ["conceptos_abstractos", "aplicacion_practica"]
        })
        variations.append(("struggling", struggling_profile))
        
        # Variación 2: Estudiante promedio
        average_profile = base_profile.copy()
        average_profile.update({
            "nivel_comprension": "intermedio",
            "fatiga_cognitiva": 0.5,
            "motivacion": 0.7,
        })
        variations.append(("average", average_profile))
        
        # Variación 3: Estudiante avanzado
        advanced_profile = base_profile.copy()
        advanced_profile.update({
            "nivel_comprension": "avanzado",
            "fatiga_cognitiva": 0.3,
            "motivacion": 0.9,
            "temas_dominados": base_profile.get("temas_dominados", []) + ["algebra_avanzada", "calculo"]
        })
        variations.append(("advanced", advanced_profile))
        
        return variations
    
    def simulate_learning_session(self, material: str, student_profile: Dict, session_params: Dict) -> Dict:
        """Simula una sesión de aprendizaje completa con métricas detalladas"""
        
        # Factores que afectan el aprendizaje
        base_comprehension = self._calculate_base_comprehension(student_profile, material)
        fatigue_factor = 1 - session_params.get("fatiga_cognitiva", 0.5)
        motivation_factor = session_params.get("motivacion", 0.7)
        time_factor = min(1.0, session_params.get("tiempo_estudio", 60) / 90)  # Óptimo en 90 min
        
        # Calcular métricas individuales
        metrics = {
            "knowledge_acquisition": min(100, base_comprehension * motivation_factor * 100),
            "retention_rate": min(100, base_comprehension * fatigue_factor * 85),
            "engagement_level": min(100, motivation_factor * time_factor * 100),
            "error_reduction": min(100, (base_comprehension * 0.8 + motivation_factor * 0.2) * 100),
            "confidence_growth": min(100, base_comprehension * motivation_factor * fatigue_factor * 120),
            "time_efficiency": min(100, (1/time_factor if time_factor > 0 else 0) * 60),
            "transfer_learning": min(100, base_comprehension * 0.6 * 100)
        }
        
        # Calcular puntuación compuesta
        composite_score = sum(
            metrics[key] * self.metrics_weights[key] 
            for key in self.metrics_weights.keys()
        )
        
        return {
            "metrics": metrics,
            "composite_score": round(composite_score, 2),
            "session_effectiveness": self._classify_effectiveness(composite_score),
            "recommendations": self._generate_recommendations(metrics, student_profile)
        }
    
    def _calculate_base_comprehension(self, profile: Dict, material: str) -> float:
        """Calcula la comprensión base basada en el perfil del estudiante"""
        nivel_mapping = {"principiante": 0.3, "intermedio": 0.6, "avanzado": 0.9}
        base_level = nivel_mapping.get(profile.get("nivel_comprension", "principiante"), 0.3)
        
        # Ajustar por conocimientos previos
        temas_dominados = len(profile.get("temas_dominados", []))
        knowledge_bonus = min(0.2, temas_dominados * 0.05)
        
        # Penalizar por áreas de dificultad
        areas_dificultad = len(profile.get("areas_dificultad", []))
        difficulty_penalty = min(0.3, areas_dificultad * 0.1)
        
        return max(0.1, min(1.0, base_level + knowledge_bonus - difficulty_penalty))
    
    def _classify_effectiveness(self, score: float) -> str:
        """Clasifica la efectividad de la sesión de aprendizaje"""
        if score >= 80:
            return "excelente"
        elif score >= 65:
            return "buena"
        elif score >= 50:
            return "regular"
        else:
            return "deficiente"
    
    def _generate_recommendations(self, metrics: Dict, profile: Dict) -> List[str]:
        """Genera recomendaciones basadas en las métricas"""
        recommendations = []
        
        if metrics["knowledge_acquisition"] < 60:
            recommendations.append("Simplificar el material y usar más ejemplos concretos")
        
        if metrics["retention_rate"] < 50:
            recommendations.append("Implementar técnicas de repetición espaciada")
        
        if metrics["engagement_level"] < 60:
            recommendations.append("Aumentar interactividad y gamificación")
        
        if metrics["error_reduction"] < 40:
            recommendations.append("Proporcionar más práctica guiada")
        
        if metrics["confidence_growth"] < 30:
            recommendations.append("Implementar logros progresivos y feedback positivo")
        
        if metrics["time_efficiency"] < 50:
            recommendations.append("Optimizar la duración y estructura de las sesiones")
        
        if metrics["transfer_learning"] < 40:
            recommendations.append("Incluir más ejemplos de aplicación práctica")
        
        return recommendations
    
    def run_experiment(self, material: str, base_profile: Dict, num_iterations: int = 100) -> Dict:
        """Ejecuta un experimento completo con múltiples simulaciones"""
        results = {
            "struggling": [],
            "average": [],
            "advanced": []
        }
        
        # Generar variaciones de perfil
        profile_variations = self.generate_student_profile_variations(base_profile)
        
        for variation_name, profile in profile_variations:
            for i in range(num_iterations):
                # Generar parámetros aleatorios para la sesión
                session_params = {
                    "fatiga_cognitiva": random.uniform(0.2, 0.9),
                    "motivacion": random.uniform(0.3, 1.0),
                    "tiempo_estudio": random.randint(30, 120),
                    "contexto_aprendizaje": random.choice(["individual", "grupal", "tutoria"])
                }
                
                # Simular sesión
                session_result = self.simulate_learning_session(material, profile, session_params)
                results[variation_name].append(session_result)
        
        # Calcular estadísticas
        experiment_summary = self._analyze_experiment_results(results)
        
        return {
            "raw_results": results,
            "summary": experiment_summary,
            "optimization_targets": self._identify_optimization_targets(experiment_summary)
        }
    
    def _analyze_experiment_results(self, results: Dict) -> Dict:
        """Analiza los resultados del experimento"""
        summary = {}
        
        for variation_name, sessions in results.items():
            if not sessions:
                continue
                
            scores = [s["composite_score"] for s in sessions]
            metrics_by_type = {}
            
            # Agregar métricas por tipo
            for metric_name in self.metrics_weights.keys():
                metric_values = [s["metrics"][metric_name] for s in sessions]
                metrics_by_type[metric_name] = {
                    "mean": np.mean(metric_values),
                    "std": np.std(metric_values),
                    "min": np.min(metric_values),
                    "max": np.max(metric_values)
                }
            
            summary[variation_name] = {
                "composite_score": {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                },
                "metrics": metrics_by_type,
                "effectiveness_distribution": self._calculate_effectiveness_distribution(sessions)
            }
        
        return summary
    
    def _calculate_effectiveness_distribution(self, sessions: List) -> Dict:
        """Calcula la distribución de efectividad"""
        effectiveness_counts = {"excelente": 0, "buena": 0, "regular": 0, "deficiente": 0}
        
        for session in sessions:
            effectiveness = session["session_effectiveness"]
            effectiveness_counts[effectiveness] += 1
        
        total = len(sessions)
        return {k: v/total for k, v in effectiveness_counts.items()}
    
    def _identify_optimization_targets(self, summary: Dict) -> Dict:
        """Identifica objetivos para optimización con metaheurísticas"""
        targets = {}
        
        for variation_name, data in summary.items():
            weak_metrics = []
            strong_metrics = []
            
            for metric_name, metric_data in data["metrics"].items():
                if metric_data["mean"] < 60:
                    weak_metrics.append(metric_name)
                elif metric_data["mean"] > 80:
                    strong_metrics.append(metric_name)
            
            targets[variation_name] = {
                "weak_metrics": weak_metrics,
                "strong_metrics": strong_metrics,
                "optimization_priority": weak_metrics[:3],  # Top 3 métricas a mejorar
                "leverage_points": strong_metrics[:2]  # Top 2 métricas a aprovechar
            }
        
        return targets
    
    async def student_simulator_chain(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Cadena principal del simulador de estudiante"""
        logger.info(f"StudentSimulator procesando: {estado.consulta_inicial}")
        
        try:
            # Determinar qué material simular
            material_a_simular = self._obtener_material_para_simular(estado)
            tipo_contenido = self._identificar_tipo_contenido(material_a_simular)
            
            # Preparar contexto del estudiante
            student_context = estado.estado_estudiante.model_dump()
            
            # Generar parámetros de simulación realistas
            simulation_params = self._generar_parametros_simulacion(student_context)
            
            prompt_data = {
                "consulta_inicial": estado.consulta_inicial,
                "material_presentado": material_a_simular[:800],  # Limitar tamaño
                "tipo_contenido": tipo_contenido,
                "nivel_comprension": student_context.get("nivel_comprension", "principiante"),
                "temas_dominados": student_context.get("temas_dominados", []),
                "areas_dificultad": student_context.get("areas_dificultad", []),
                "historial_errores": student_context.get("historial_errores", []),
                "estilo_aprendizaje": simulation_params["estilo_aprendizaje"],
                "fatiga_cognitiva": simulation_params["fatiga_cognitiva"],
                "motivacion": simulation_params["motivacion"],
                "tiempo_estudio": simulation_params["tiempo_estudio"],
                "contexto_aprendizaje": simulation_params["contexto_aprendizaje"]
            }
            
            # Ejecutar simulación con múltiples intentos
            simulacion = await self._ejecutar_simulacion_con_fallback(prompt_data)
            
            # Ejecutar experimento completo si es necesario
            if estado.estado_actual == "requiere_experimentacion":
                experimento = self.run_experiment(
                    material_a_simular, 
                    student_context,
                    num_iterations=50  # Reducido para performance
                )
                simulacion["experiment_results"] = experimento
            
            # Actualizar estado
            estado.respuesta_student_simulator = self._format_simulation_output(simulacion)
            estado.estado_actual = "student_simulator_completado"
            
            # Agregar al historial
            estado.chat_history.append({
                "role": "student_simulator",
                "content": estado.respuesta_student_simulator,
                "metadata": {
                    "simulation_type": tipo_contenido,
                    "student_profile": student_context["nivel_comprension"],
                    "learning_metrics": simulacion.get("learning_metrics", {}),
                    "optimization_potential": simulacion.get("experiment_results", {}).get("optimization_targets", {}),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            logger.info(f"✅ StudentSimulator completado con métricas de aprendizaje")
            return estado
            
        except Exception as e:
            logger.error(f"💥 Error crítico en student simulator: {e}")
            import traceback
            traceback.print_exc()
            
            # Respuesta de emergencia
            estado.respuesta_student_simulator = self._create_fallback_simulation(estado)
            estado.estado_actual = "student_simulator_completado"
            
            return estado
    
    def _obtener_material_para_simular(self, estado: EstadoConversacion) -> str:
        """Obtiene el material que debe simular el estudiante"""
        if estado.respuesta_math_expert:
            return estado.respuesta_math_expert
        elif estado.respuesta_exam_creator:
            return estado.respuesta_exam_creator
        elif estado.contexto_recuperado:
            return str(estado.contexto_recuperado[0].get("content", ""))
        else:
            return estado.consulta_inicial
    
    def _identificar_tipo_contenido(self, material: str) -> str:
        """Identifica el tipo de contenido para personalizar la simulación"""
        material_lower = material.lower()
        
        if any(word in material_lower for word in ["examen", "quiz", "test", "pregunta"]):
            return "evaluacion"
        elif any(word in material_lower for word in ["explicacion", "concepto", "teorema", "definicion"]):
            return "conceptual"
        elif any(word in material_lower for word in ["problema", "ejercicio", "resuelve", "calcula"]):
            return "procedimental"
        else:
            return "mixto"
    
    def _generar_parametros_simulacion(self, student_context: Dict) -> Dict:
        """Genera parámetros realistas para la simulación"""
        nivel = student_context.get("nivel_comprension", "principiante")
        
        # Parámetros base por nivel
        if nivel == "principiante":
            base_params = {
                "fatiga_cognitiva": random.uniform(0.5, 0.8),
                "motivacion": random.uniform(0.4, 0.7),
                "tiempo_estudio": random.randint(20, 45)
            }
        elif nivel == "intermedio":
            base_params = {
                "fatiga_cognitiva": random.uniform(0.3, 0.6),
                "motivacion": random.uniform(0.6, 0.8),
                "tiempo_estudio": random.randint(30, 60)
            }
        else:  # avanzado
            base_params = {
                "fatiga_cognitiva": random.uniform(0.2, 0.5),
                "motivacion": random.uniform(0.7, 0.9),
                "tiempo_estudio": random.randint(45, 90)
            }
        
        base_params.update({
            "estilo_aprendizaje": random.choice(["visual", "auditivo", "kinestesico", "mixto"]),
            "contexto_aprendizaje": random.choice(["individual", "grupal", "tutoria"])
        })
        
        return base_params
    
    async def _ejecutar_simulacion_con_fallback(self, prompt_data: Dict) -> Dict:
        """Ejecuta la simulación con múltiples estrategias de fallback"""
        try:
            # Intentar structured output
            formatted_prompt = self.prompt.format(**prompt_data)
            respuesta = await self.llm_structured.ainvoke(formatted_prompt)
            
            if isinstance(respuesta, dict):
                return respuesta
            elif hasattr(respuesta, 'model_dump'):
                return respuesta.model_dump()
            else:
                return self._convert_to_simulation_dict(respuesta)
                
        except Exception as e:
            logger.warning(f"⚠️ Structured output falló, usando fallback: {e}")
            return self._create_realistic_simulation_fallback(prompt_data)
    
    def _convert_to_simulation_dict(self, respuesta) -> Dict:
        """Convierte respuesta a formato de simulación"""
        try:
            if hasattr(respuesta, '__dict__'):
                return respuesta.__dict__
            else:
                return {
                    "student_answer": str(respuesta),
                    "confidence_level": "media",
                    "common_mistakes": ["Error de comprensión general"],
                    "reasoning_process": "Proceso simulado básico",
                    "areas_of_confusion": ["Conceptos avanzados"],
                    "learning_metrics": self._generate_default_metrics(),
                    "cognitive_load": "medio",
                    "motivation_change": 0,
                    "predicted_performance": 60
                }
        except:
            return self._create_realistic_simulation_fallback({})
    
    def _generate_default_metrics(self) -> Dict:
        """Genera métricas por defecto realistas"""
        return {
            "knowledge_acquisition": random.randint(50, 80),
            "retention_rate": random.randint(40, 70),
            "engagement_level": random.randint(60, 85),
            "error_reduction": random.randint(30, 60),
            "confidence_growth": random.randint(20, 50),
            "time_efficiency": random.randint(50, 80),
            "transfer_learning": random.randint(30, 60)
        }
    
    def _create_realistic_simulation_fallback(self, prompt_data: Dict) -> Dict:
        """Crea una simulación realista de fallback"""
        nivel = prompt_data.get("nivel_comprension", "principiante")
        
        # Simular respuesta basada en nivel
        if nivel == "principiante":
            return {
                "student_answer": "Creo que entiendo parte del concepto, pero me confundo en los pasos más complejos.",
                "confidence_level": "baja",
                "common_mistakes": ["Confusión en terminología", "Errores de procedimiento básico"],
                "reasoning_process": "Intento seguir los pasos pero me pierdo en la aplicación práctica",
                "areas_of_confusion": ["Conceptos abstractos", "Aplicación de fórmulas"],
                "learning_metrics": {
                    "knowledge_acquisition": random.randint(30, 60),
                    "retention_rate": random.randint(25, 50),
                    "engagement_level": random.randint(40, 70),
                    "error_reduction": random.randint(20, 40),
                    "confidence_growth": random.randint(10, 30),
                    "time_efficiency": random.randint(40, 65),
                    "transfer_learning": random.randint(20, 45)
                },
                "cognitive_load": "alto",
                "motivation_change": -2,
                "predicted_performance": random.randint(35, 55)
            }
        elif nivel == "intermedio":
            return {
                "student_answer": "Comprendo la mayoría de conceptos, aunque tengo algunas dudas en la aplicación avanzada.",
                "confidence_level": "media",
                "common_mistakes": ["Errores de precisión", "Confusión en casos especiales"],
                "reasoning_process": "Sigo el proceso lógico pero a veces dudo en los detalles",
                "areas_of_confusion": ["Casos límite", "Aplicaciones complejas"],
                "learning_metrics": {
                    "knowledge_acquisition": random.randint(60, 80),
                    "retention_rate": random.randint(50, 75),
                    "engagement_level": random.randint(65, 85),
                    "error_reduction": random.randint(40, 65),
                    "confidence_growth": random.randint(25, 45),
                    "time_efficiency": random.randint(60, 80),
                    "transfer_learning": random.randint(40, 65)
                },
                "cognitive_load": "medio",
                "motivation_change": 1,
                "predicted_performance": random.randint(60, 75)
            }
        else:  # avanzado
            return {
                "student_answer": "Entiendo el concepto claramente y puedo aplicarlo en diversos contextos.",
                "confidence_level": "alta",
                "common_mistakes": ["Errores de cálculo menor", "Optimizaciones innecesarias"],
                "reasoning_process": "Comprendo la lógica subyacente y puedo extrapolar",
                "areas_of_confusion": ["Aplicaciones muy específicas"],
                "learning_metrics": {
                    "knowledge_acquisition": random.randint(80, 95),
                    "retention_rate": random.randint(75, 90),
                    "engagement_level": random.randint(80, 95),
                    "error_reduction": random.randint(65, 85),
                    "confidence_growth": random.randint(35, 55),
                    "time_efficiency": random.randint(75, 90),
                    "transfer_learning": random.randint(60, 85)
                },
                "cognitive_load": "bajo",
                "motivation_change": 3,
                "predicted_performance": random.randint(80, 95)
            }
    
    def _create_fallback_simulation(self, estado: EstadoConversacion) -> str:
        """Crea simulación de emergencia"""
        return f"""
                ## Simulación de Estudiante - {estado.estado_estudiante.nivel_comprension.title()}

                **Respuesta Simulada:**
                "Basándome en mi nivel actual, creo que {random.choice(['entiendo parcialmente', 'comprendo la mayoría', 'domino'])} el concepto presentado."

                **Métricas de Aprendizaje:**
                - Adquisición de conocimiento: {random.randint(40, 80)}%
                - Retención estimada: {random.randint(35, 70)}%
                - Nivel de engagement: {random.randint(50, 85)}%

                **Recomendaciones:**
                - Ajustar nivel de dificultad según perfil del estudiante
                - Implementar más ejemplos prácticos
                - Considerar refuerzo en áreas débiles
        """
    
    def _format_simulation_output(self, simulacion: Dict) -> str:
        """Formatea la salida de la simulación para mostrar al usuario"""
        output = f"# Simulación de Estudiante\n\n"
        
        output += f"**Respuesta del Estudiante Simulado:**\n"
        output += f'"{simulacion.get("student_answer", "Sin respuesta")}"]\n\n'
        
        output += f"**Nivel de Confianza:** {simulacion.get('confidence_level', 'media').title()}\n\n"
        
        output += f"**Proceso de Razonamiento:**\n"
        output += f"{simulacion.get('reasoning_process', 'Proceso no especificado')}\n\n"
        
        # Métricas de aprendizaje
        metrics = simulacion.get("learning_metrics", {})
        if metrics:
            output += f"## Métricas de Aprendizaje\n\n"
            for metric, value in metrics.items():
                metric_name = metric.replace("_", " ").title()
                output += f"- **{metric_name}:** {value}%\n"
            output += "\n"
        
        # Errores comunes
        mistakes = simulacion.get("common_mistakes", [])
        if mistakes:
            output += f"**Errores Comunes Identificados:**\n"
            for mistake in mistakes:
                output += f"- {mistake}\n"
            output += "\n"
        
        # Áreas de confusión
        confusion_areas = simulacion.get("areas_of_confusion", [])
        if confusion_areas:
            output += f"**Áreas de Confusión:**\n"
            for area in confusion_areas:
                output += f"- {area}\n"
            output += "\n"
        
        # Resultados experimentales si existen
        if "experiment_results" in simulacion:
            exp_results = simulacion["experiment_results"]
            output += f"## Resultados de Experimentación\n\n"
            
            summary = exp_results.get("summary", {})
            for variation, data in summary.items():
                output += f"**Perfil {variation.title()}:**\n"
                output += f"- Puntuación promedio: {data['composite_score']['mean']:.1f}\n"
                output += f"- Distribución de efectividad: {data['effectiveness_distribution']}\n"
            
            # Objetivos de optimización
            opt_targets = exp_results.get("optimization_targets", {})
            if opt_targets:
                output += f"\n**Objetivos para Optimización con Metaheurísticas:**\n"
                for variation, targets in opt_targets.items():
                    output += f"- {variation.title()}: Mejorar {', '.join(targets['optimization_priority'])}\n"
        
        output += "\n---\n\n*Simulación generada por StudentSimulatorAgent para optimización pedagógica*"
        
        return output