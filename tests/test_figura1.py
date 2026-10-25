import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problem import MaintenanceProblem

def test_figura_1():
    print("=== TESTANDO EXEMPLO DA FIGURA 1 (n=4) ===")
    
    fig1_path = os.path.join("data", "figura1.txt")
    problem = MaintenanceProblem(fig1_path)

    # Rota da Figura 1: (1, 4, 2, 3)
    pi_gulosa = [1, 4, 2, 3]
    h_gulosa = problem.h(pi_gulosa)
    print(f"Rota (1, 4, 2, 3) h(pi): {h_gulosa} | Esperado: 39")
    assert h_gulosa == 39, f"ERRO: Rota (1,4,2,3) deu {h_gulosa}, esperado 39"

    # Rota Ótima da Figura 1: (1, 4, 3, 2)
    pi_otima = [1, 4, 3, 2]
    h_otima = problem.h(pi_otima)
    print(f"Rota Ótima (1, 4, 3, 2) h(pi*): {h_otima} | Esperado: 22")
    assert h_otima == 22, f"ERRO: Rota Ótima (1,4,3,2) deu {h_otima}, esperado 22"

    print("✓ Validação da Figura 1 aprovada com sucesso!\n")

if __name__ == "__main__":
    test_figura_1()