import asyncio
import json
import time
import os
from pathlib import Path
from agents.specialised_agents.math_agent import MathExpert
from agents.specialised_agents.exam_creator_agent import ExamCreatorAgent
from agents.specialised_agents.planning_agent import PlanningAgent
from agents.specialised_agents.evaluator_agent import EvaluatorAgent
from agents.dto_s.agent_state import EstadoConversacion, EstudianteProfile, BDIState
from generator.llm_provider import MistralLLMProvider

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..' , '..'))

AGENT_MAP = {
    "math_expert": (MathExpert, "math_expert_chain"),
    "exam_creator": (ExamCreatorAgent, "exam_creator_chain"),
    "planning": (PlanningAgent, "plannig_chain"),
    "evaluator": (EvaluatorAgent, "evaluator_chain"),
}

def load_test_cases(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]

async def medir_tiempo_agente(llm, agent_name, consulta, context_state=None):
    AgentClass, method_name = AGENT_MAP[agent_name]
    agente = AgentClass(llm)
    estado = EstadoConversacion(
        consulta_inicial=consulta,
        estado_actual="inicio",
        estado_estudiante=EstudianteProfile(),
        bdi_state=BDIState()
    )
    if context_state:
        for k, v in context_state.items():
            setattr(estado, k, v)
    chain_method = getattr(agente, method_name)
    start = time.perf_counter()
    await chain_method(estado)
    end = time.perf_counter()
    return end - start

async def main():
    
    llm = MistralLLMProvider()
    DATA_ROOT = Path(project_root) / ".data"
    TIME_EXP_ROOT = DATA_ROOT / "experiments" / "timer"
    test_cases_path = TIME_EXP_ROOT / "test_cases.json"    
    results_path = Path(__file__).parent / "results.json"
    test_cases = load_test_cases(test_cases_path)
    results = []

    for case in test_cases:
        agent = case["agent"]
        consulta = case["consulta"]
        context_state = case.get("context_state")
        tiempo = await medir_tiempo_agente(llm, agent, consulta, context_state)
        result = {
            "id": case["id"],
            "consulta": consulta,
            "agent": agent,
            "difficulty": case["difficulty"],
            "category": case["category"],
            "response_time": tiempo
        }
        print(f"{case['id']} ({agent}): {tiempo:.3f} s")
        results.append(result)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {results_path}")

if __name__ == "__main__":
    asyncio.run(main())