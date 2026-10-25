import sys
import os

# Adiciona a raiz do projeto ao PYTHONPATH para reconhecer o pacote 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problem import MaintenanceProblem
from src.evaluator import Evaluator, BudgetExhaustedException

def run_verification():
    print("=== EXECUTANDO TESTE DE VALIDAÇÃO (T1) ===")
    
    # 1. Carrega a instância pequena
    data_path = os.path.join("data", "manutencao_pequena.txt")
    problem = MaintenanceProblem(data_path)
    
    # 2. Testa a Permutação Identidade: (1, 2, 3, ..., 12)
    identity_pi = list(range(1, problem.n + 1))
    h_identity = problem.h(identity_pi)
    
    print(f"Instância: manutencao_pequena.txt (n={problem.n})")
    print(f"Permutação Identidade h(pi): {h_identity} | Esperado: 4366")
    
    assert h_identity == 4366, f"ERRO: h(identidade) retornou {h_identity}, esperado 4366!"
    print("✓ Sucesso: Fitness da permutação identidade validado!")

    # 3. Testa a restrição do Budget (Evaluator)
    evaluator = Evaluator(problem, max_evaluations=10)
    print("\n=== TESTANDO GERENCIADOR DE ORÇAMENTO (BUDGET) ===")

    ok = 0
    try:
        for _ in range(15):
            evaluator.evaluate(identity_pi)
            ok += 1
        print("✗ ERRO: O Evaluator permitiu ultrapassar o limite!")
    except BudgetExhaustedException:
        print("✓ BudgetExhaustedException lançada após o limite.")

    assert ok == 10, f"ERRO: {ok} avaliações bem-sucedidas, esperado 10"
    assert evaluator.evaluations_count == 10, \
        f"ERRO: contador em {evaluator.evaluations_count}, esperado 10"
    print("✓ Sucesso: exatamente 10 avaliações permitidas e contador parou em 10!")

if __name__ == "__main__":
    run_verification()