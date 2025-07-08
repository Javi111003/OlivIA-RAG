import asyncio
import os
import json
from pathlib import Path
from agents.supervisor_agent import SupervisorAgent
from agents.dto_s.agent_state import EstadoConversacion, EstudianteProfile, BDIState
from generator.llm_provider import MistralLLMProvider

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..' , '..'))

def load_test_cases(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]

# --- Ejecutar experimento ---
async def run_supervisor_experiment(test_cases, results_path):
    llm = MistralLLMProvider()
    supervisor = SupervisorAgent(llm)
    results = []

    for case in test_cases:
        estado = EstadoConversacion(
            consulta_inicial=case["consulta"],
            estado_actual="inicio",
            estado_estudiante=EstudianteProfile(),
            bdi_state=BDIState()
        )
        if "context_state" in case and case["context_state"]:
            for k, v in case["context_state"].items():
                setattr(estado, k, v)
        estado_result = await supervisor.supervisor_chain(estado)
        predicted_agent = estado_result.tipo_ayuda_necesaria
        results.append({
            "id": case["id"],
            "consulta": case["consulta"],
            "expected_agent": case["expected_agent"],
            "predicted_agent": predicted_agent,
            "difficulty": case["difficulty"],
            "category": case["category"],
            "correct": predicted_agent == case["expected_agent"]
        })
        print(f"{case['id']}: esperado={case['expected_agent']} | predicho={predicted_agent} | {'✅' if predicted_agent == case['expected_agent'] else '❌'}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {results_path}")

if __name__ == "__main__":

    DATA_ROOT = Path(project_root) / ".data"
    SUPERVISOR_EXP_ROOT = DATA_ROOT / "experiments" / "supervisor"
    test_cases_path = SUPERVISOR_EXP_ROOT / "test_cases.json"
    results_path = Path(__file__).parent / "results.json"
    test_cases = load_test_cases(test_cases_path)
    asyncio.run(run_supervisor_experiment(test_cases, results_path))