from datetime import datetime
from logging import getLogger
from langgraph.graph import StateGraph, END
from  agents.dto_s.agent_state import EstadoConversacion
from  generator.llm_provider import MistralLLMProvider
from  agents.supervisor_agent import SupervisorAgent
from  agents.specialised_agents.math_agent import MathExpert
from  agents.specialised_agents.exam_creator_agent import ExamCreatorAgent
from  agents.specialised_agents.evaluator_agent import EvaluatorAgent
from  retriever.dense_retriever import DenseRetriever
from  embedding_models.embedding_generator import EmbeddingGenerator
from agents.specialised_agents.planning_agent import PlanningAgent
logger = getLogger(__name__)

class AgenticPipeline:
    """Pipeline de agentes con arquitectura BDI y personalización pedagógica"""
    
    def __init__(self, llm : MistralLLMProvider, config=None):
        self.llm = llm
        self.config = config or {}
        
        self.retriever = DenseRetriever()
        self.supervisor = SupervisorAgent(llm)
        self.supervisor_chain = self.supervisor.supervisor_chain
        self.supervisor_router = self.supervisor.supervisor_router
        self.math_expert = MathExpert(llm)
        self.math_expert_chain = self.math_expert.math_expert_chain
        self.exam_creator = ExamCreatorAgent(llm)
        self.exam_creator_chain = self.exam_creator.exam_creator_chain
        # self.student_simulator = crear_student_simulator(llm)
        self.evaluator = EvaluatorAgent(llm)
        self.evaluator_chain = self.evaluator.evaluator_chain
        self.planning = PlanningAgent(llm)
        self.planning_chain = self.planning.plannig_chain
        
        self._build_graph()
    
    def _build_graph(self):
        """Construye el grafo de agentes con enrutamiento inteligente"""
        workflow = StateGraph(EstadoConversacion)
        
        # Añadir nodos
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("supervisor", self.supervisor_chain)
        workflow.add_node("math_expert", self.math_expert_chain)
        workflow.add_node("exam_creator", self.exam_creator_chain)
        workflow.add_node("planning", self.planning_chain)
        # workflow.add_node("student_simulator", self.student_simulator)
        workflow.add_node("evaluator", self.evaluator_chain)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Definir flujo
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            self.supervisor_router,
            {
                "math_expert": "math_expert",
                "exam_creator": "exam_creator",
                #"student_simulator": "student_simulator",
                "evaluator": "evaluator",
                "planning" : "planning",
                "FINISH": "finalizer"
            }
        )
        
        workflow.add_edge("math_expert", "supervisor")
        workflow.add_edge("exam_creator", "supervisor")
        workflow.add_edge("planning", "supervisor")
        # workflow.add_edge("student_simulator", "supervisor")
        workflow.add_edge("evaluator", "supervisor")
        
        workflow.add_edge("finalizer", END)
        
        self.graph = workflow.compile()
        self.visualize_graph()
    
    async def _retriever_node(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Nodo de recuperación con DenseRetriever real"""
        try:
            logger.info(f"🔍 RETRIEVER: Procesando consulta: {estado.consulta_inicial}")
            
            if not any(msg.get("role") == "user" and msg.get("content") == estado.consulta_inicial 
                for msg in estado.chat_history):
                estado.chat_history.append({
                    "role": "user",
                    "content": estado.consulta_inicial,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "is_initial_query": True
                    }
                })
            embedding_generator = EmbeddingGenerator()
            query_embedding = embedding_generator.generate_embedding(estado.consulta_inicial)
            logger.info(f"📊 RETRIEVER: Embedding generado, dimensión: {query_embedding.shape}")
            
            resultados = self.retriever.retrieve(query_embedding, top_k=3)
            
            estado.contexto_recuperado = [
                {"content": doc, "score": float(score)} 
                for doc, score in resultados
            ]
            print(estado.contexto_recuperado)
            estado.estado_actual = "retriever_completado"
            
            logger.info(f"✅ RETRIEVER: Recuperados {len(resultados)} documentos")
            for i, (doc, score) in enumerate(resultados):
                logger.info(f"   📄 Doc {i+1}: Score {score:.3f} - {doc[:100]}...")
            
            return estado
            
        except Exception as e:
            logger.error(f"💥 RETRIEVER ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback a simulación si falla
            estado.contexto_recuperado = [
                {"content": "Documento matemático de fallback 1", "score": 0.5},
                {"content": "Documento matemático de fallback 2", "score": 0.4}
            ]
            estado.estado_actual = "retriever_completado"
            return estado
    
    async def _finalizer_node(self, estado: EstadoConversacion) -> EstadoConversacion:
        """Nodo finalizador que compila la respuesta completa"""
        logger.info(f"🏁 Finalizador ejecutándose - Estado: {estado.estado_actual}")
        
        # Priorizar respuestas disponibles
        if estado.respuesta_math_expert:
            estado.respuesta_final = estado.respuesta_math_expert
            logger.info("✅ Finalizando con respuesta de math_expert")
        elif estado.respuesta_exam_creator:
            estado.respuesta_final = estado.respuesta_exam_creator
            logger.info("✅ Finalizando con respuesta de exam_creator")
        elif estado.respuesta_planning:
            estado.respuesta_final = estado.respuesta_planning
            logger.info("✅ Finalizando con respuesta de planning")
        else:
            estado.respuesta_final = "No se pudo generar una respuesta adecuada."
            logger.info("⚠️ Finalizando sin respuesta específica")
        
        estado.estado_actual = "FINISH"
        estado.necesita_crawler = False
        
        logger.info(f"🏁 Pipeline finalizado. Respuesta: {estado.respuesta_final[:100]}...")
        
        return estado
    
    def visualize_graph(self):
        """Visualiza el grafo y guarda/muestra el resultado"""
        try:
            # Generar el diagrama Mermaid
            mermaid_code = self.graph.get_graph().draw_mermaid()
            
            # Guardar en archivo
            with open("graph_visualization.md", "w", encoding="utf-8") as f:
                f.write("# Grafo del Pipeline de Agentes\n\n")
                f.write("```mermaid\n")
                f.write(mermaid_code)
                f.write("\n```")
            
            print("=" * 60)
            print("VISUALIZACIÓN DEL GRAFO GENERADA")
            print("=" * 60)
            print("\n📁 Archivo guardado: graph_visualization.md")
            print("\n🔗 Código Mermaid:\n")
            print(mermaid_code)
            print("\n" + "=" * 60)
            
            # información del grafo
            print(f"📊 Nodos del grafo: {list(self.graph.get_graph().nodes)}")
            print(f"🔗 Aristas del grafo: {list(self.graph.get_graph().edges)}")
            
            return mermaid_code
            
        except Exception as e:
            print(f"❌ Error visualizando grafo: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def debug_resultado(self, resultado):
        """Función de debug para entender el resultado de LangGraph"""
        print("\n" + "="*50)
        print("🔍 DEBUGGING RESULTADO")
        print("="*50)
        
        chat_history = resultado.get('chat_history', [])
        evaluator_messages = [msg for msg in chat_history if msg.get('role') == 'evaluator']
        
        print(f"🤖 EVALUADOR:")
        if evaluator_messages:
            print(f"   ✅ Ejecutado {len(evaluator_messages)} veces")
            for i, msg in enumerate(evaluator_messages):
                print(f"   📋 Evaluación {i+1}: {msg.get('content', 'Sin contenido')}")
                if 'metadata' in msg:
                    metadata = msg['metadata']
                    if 'evaluation' in metadata:
                        eval_data = metadata['evaluation']
                        print(f"      - Calidad: {eval_data.get('overall_quality', 'N/A')}")
                        print(f"      - Suficiente: {eval_data.get('is_sufficient', 'N/A')}")
                        print(f"      - Puntuaciones: {eval_data.get('scores', {})}")
        else:
            print(f"   ❌ NO SE EJECUTÓ")
        print(f"Tipo: {type(resultado)}")
        print(f"Dir: {dir(resultado)}")
        
        if hasattr(resultado, 'keys'):
            print(f"Claves: {list(resultado.keys())}")
            for key in resultado.keys():
                print(f"  {key}: {type(resultado[key])} = {resultado[key]}")
        
        if hasattr(resultado, '__dict__'):
            print(f"Dict: {resultado.__dict__}")
        
        print("="*50 + "\n")
        return resultado
    
    async def run(self, consulta: str) -> str:
        """Ejecuta el pipeline completo"""
        print(f"\n🚀 Iniciando pipeline con consulta: '{consulta}'")
        
        # Estado inicial
        estado_inicial = EstadoConversacion(
            consulta_inicial=consulta,
            estado_actual="inicio"
        )
        
        try:
            # Ejecutar el grafo
            resultado = await self.graph.ainvoke(estado_inicial, config=self.config)
            
            self.debug_resultado(resultado)

            print(f"✅ Pipeline completado exitosamente")
            return resultado['respuesta_final']
            
        except Exception as e:
            print(f"❌ Error ejecutando pipeline: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
        
if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        print("🔧 Inicializando pipeline...")
        
        try:
            llm = MistralLLMProvider()
            pipeline = AgenticPipeline(llm)
            print("\n📋 Ejecutando consulta de prueba...")
            resultado = await pipeline.run("Explícame el teorema de las tres perpendiculares")
            
            print(f"\n📝 Resultado final: {resultado}")
            
            print("\n📋 Ejecutando consulta 2 de prueba...")
            resultado = await pipeline.run("Hablame sobre geometría analítica")
            
            print(f"\n📝 Resultado final: {resultado}")
            
            
        except Exception as e:
            print(f"❌ Error en test: {e}")
            import traceback
            traceback.print_exc()
    
    # Ejecutar el test
    asyncio.run(test_pipeline())